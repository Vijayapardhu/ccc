#!/usr/bin/env bash
# Fetch the ONNX model weights.
#
# Weights are not vendored: they are ~500MB of binaries and InsightFace's
# licence is not ours to redistribute. Fetch them into ./models on every node
# that runs inference (capture workers and the enrollment station). The API
# and identity services do not need them — they never touch a pixel.
#
# Idempotent. A file under 1MB is treated as missing, because a truncated
# download from an interrupted run otherwise persists and fails much later,
# inside ONNX Runtime, as an opaque error.

set -euo pipefail

MODELS_DIR="${1:-models}"
MIN_BYTES=1048576
CACHE="$(mktemp -d)"
trap 'rm -rf "$CACHE"' EXIT

mkdir -p "$MODELS_DIR"

size_of() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }

have() {
  local dest="$MODELS_DIR/$1"
  [[ -f "$dest" ]] && [[ "$(size_of "$dest")" -ge $MIN_BYTES ]]
}

# extract <archive-url> <member-name> <filename>
extract() {
  local url="$1" member="$2" out="$3"
  local dest="$MODELS_DIR/$out"

  if have "$out"; then
    echo "  ok      $out"
    return
  fi

  echo "  fetch   $out  (from $(basename "$url"))"
  local zip="$CACHE/$(basename "$url")"
  [[ -f "$zip" ]] || curl --fail --location --silent --show-error \
                          --retry 3 --retry-delay 2 -o "$zip" "$url"

  if ! unzip -o -j "$zip" "$member" -d "$MODELS_DIR" >/dev/null; then
    echo "error: $member not found in $(basename "$url")" >&2
    echo "      list the archive with: unzip -l $zip" >&2
    exit 1
  fi
  echo "  ok      $out"
}

echo "Fetching models into $MODELS_DIR"

# antelopev2 carries both the small dense-crowd detector and the stronger
# embedder, so one download covers the whole live pipeline.
ANTELOPE="https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
extract "$ANTELOPE" "scrfd_2.5g.onnx"    "scrfd_2.5g.onnx"    # live detection
extract "$ANTELOPE" "glintr100k.onnx"    "glintr100k.onnx"    # embedding
# Enrollment detector: 3x slower, materially better on small off-centre faces.
extract "$ANTELOPE" "scrfd_10g.onnx"     "scrfd_10g.onnx"     # enrollment

# CPU fallback for dev boxes and the enrollment station.
BUFFALO="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
extract "$BUFFALO"  "mobilefacenet.onnx" "mobilefacenet.onnx" # CPU embedding, 128-d

echo
echo "Done. Verify with:"
echo "  campus validate --config configs/system.yaml --cameras configs/cameras.yaml"
echo
echo "Note: mobilefacenet outputs 128-d, glintr100k 512-d. The loader refuses a"
echo "mismatch rather than silently mixing dimensionalities in one gallery —"
echo "if you switch, re-enroll the affected students."
