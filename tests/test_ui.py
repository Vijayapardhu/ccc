"""The inspection dashboard.

The pure logic is tested directly rather than through an ASGI client, because
the thing that broke was a serialisation bug in one function and there is no
reason that should require a live server to catch.

The HTTP tests are kept but skipped when the installed starlette/httpx pair is
incompatible — starlette < 0.28's TestClient passes an ``app=`` kwarg that
httpx >= 0.28 removed, which fails at construction with an unrelated-looking
``TypeError`` that has nothing to do with this project.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from campus.gallery import GalleryFile
from campus.ui import app as ui_app
from campus.ui.app import _page_meta, roster_rows


@pytest.fixture
def gallery() -> GalleryFile:
    return GalleryFile(
        student_ids=["25B11CS001", "26B21CS142", "25B11EE003"],
        vectors=np.eye(3, 512, dtype=np.float32),
        dim=512,
        photos_per_student={"25B11CS001": 1, "26B21CS142": 1, "25B11EE003": 3},
        built_at=1000.0,
        source="test",
        model="test.onnx",
    )


def _client_compatible() -> bool:
    try:
        from fastapi.testclient import TestClient  # noqa: PLC0415

        from campus.ui.app import api  # noqa: PLC0415

        TestClient(api)
    except TypeError:
        return False
    except Exception:  # noqa: BLE001
        return True
    return True


requires_client = pytest.mark.skipif(
    not _client_compatible(),
    reason="starlette/httpx TestClient incompatibility in this environment",
)


class TestPageMeta:
    """Regression: a 500 on `/` takes out the one page you need to debug."""

    def test_serialises_clean_metadata(self, gallery: GalleryFile):
        ui_app.configure({}, {"gallery": gallery.summary(), "_gallery_file": gallery})
        try:
            assert json.loads(_page_meta())["gallery"]["students"] == 3
        finally:
            ui_app.configure({}, {})

    def test_drops_non_serializable_fields(self, gallery: GalleryFile):
        """A live object in the metadata must not raise here."""
        ui_app.configure({}, {"gallery": gallery.summary(), "_gallery_file": gallery})
        ui_app.meta["_oops"] = object()
        try:
            payload = json.loads(_page_meta())
            assert "_oops" not in payload
        finally:
            ui_app.configure({}, {})

    def test_gallery_object_never_reaches_the_payload(self, gallery: GalleryFile):
        """The bug that started this: the GalleryFile dataclass was in `meta`,
        which is serialised into the HTML, so every page load 500'd."""
        ui_app.configure({}, {"gallery": gallery.summary(), "_gallery_file": gallery})
        try:
            assert "_gallery_file" not in json.loads(_page_meta())
        finally:
            ui_app.configure({}, {})


class TestRosterLogic:
    def test_lists_every_student(self, gallery: GalleryFile):
        r = roster_rows(gallery)
        assert r["total"] == 3
        assert {s["student_id"] for s in r["students"]} == set(gallery.student_ids)

    def test_reports_photo_counts(self, gallery: GalleryFile):
        by_id = {s["student_id"]: s["photos"] for s in roster_rows(gallery)["students"]}
        assert by_id["25B11EE003"] == 3
        assert by_id["26B21CS142"] == 1

    def test_search_is_case_insensitive(self, gallery: GalleryFile):
        r = roster_rows(gallery, "26b21cs142")
        assert r["matched"] == 1
        assert r["students"][0]["student_id"] == "26B21CS142"

    def test_search_with_no_match(self, gallery: GalleryFile):
        """The case that matters: confirming an id is NOT enrolled."""
        r = roster_rows(gallery, "does-not-exist")
        assert r["matched"] == 0
        assert r["students"] == []
        assert r["total"] == 3, "total must not collapse when filtering"

    def test_prefix_and_substring_search(self, gallery: GalleryFile):
        assert roster_rows(gallery, "25B11")["matched"] == 2
        assert roster_rows(gallery, "EE")["matched"] == 1

    def test_blank_query_returns_all(self, gallery: GalleryFile):
        assert roster_rows(gallery, "   ")["matched"] == 3

    def test_limit_is_respected(self, gallery: GalleryFile):
        assert len(roster_rows(gallery, "", limit=2)["students"]) == 2

    def test_no_gallery_returns_empty(self):
        r = roster_rows(None)
        assert r == {"total": 0, "matched": 0, "query": "", "students": []}


@requires_client
class TestHttp:
    def test_index_renders(self, gallery: GalleryFile):
        from fastapi.testclient import TestClient  # noqa: PLC0415

        ui_app.configure({}, {"gallery": gallery.summary(), "_gallery_file": gallery})
        try:
            r = TestClient(ui_app.api).get("/")
            assert r.status_code == 200, r.text[:300]
            assert "Campus pipeline inspector" in r.text
        finally:
            ui_app.configure({}, {})

    def test_roster_endpoint(self, gallery: GalleryFile):
        from fastapi.testclient import TestClient  # noqa: PLC0415

        ui_app.configure({}, {"gallery": gallery.summary(), "_gallery_file": gallery})
        try:
            r = TestClient(ui_app.api).get("/api/gallery/students?q=26B21CS142")
            assert r.status_code == 200
            assert r.json()["matched"] == 1
        finally:
            ui_app.configure({}, {})

    def test_state_shape(self, gallery: GalleryFile):
        from fastapi.testclient import TestClient  # noqa: PLC0415

        ui_app.configure({}, {"gallery": gallery.summary(), "_gallery_file": gallery})
        try:
            body = TestClient(ui_app.api).get("/api/state").json()
            assert set(body) == {"meta", "cameras", "events"}
        finally:
            ui_app.configure({}, {})
