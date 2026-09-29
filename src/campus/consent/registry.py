"""DPDP Act 2023 consent and purpose enforcement.

The DPDP Act treats a face embedding as personal data and imposes three
obligations that have direct architectural consequences:

1. **Consent** (s.6) — processing needs a lawful basis, and for a campus
   deployment the realistic one is explicit consent, not legitimate interest.
   Consent must be specific, informed, and withdrawable at any time.
2. **Purpose limitation** (s.5-6) — a consent given for "library access
   control" does not cover "attendance analytics". Consent is stored per
   *purpose*, not as a single boolean.
3. **Erasure** (s.12) — a withdrawal must actually remove the vectors, not
   just set a flag.

This is the one place where the system is required to be able to say "no" to
itself. Every identity resolution passes through :meth:`ConsentRegistry.check`
before a student_id is attached, and the identity service does not cache that
decision across requests — a consent revoked mid-semester takes effect on the
next frame, not the next deploy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from campus.types import ConsentDecision


class Purpose(StrEnum):
    """Discrete purposes. Each needs its own consent checkbox.

    Purpose limitation is the requirement most often botched, so the enum is
    closed: adding a purpose is a code change that forces someone to think
    about the consent text, rather than a string that quietly defaults to
    allowed.
    """

    ATTENDANCE = "attendance"
    SAFETY = "safety"
    ACCESS_CONTROL = "access_control"
    SPACE_UTILISATION = "space_utilisation"
    RESEARCH = "research"


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    student_id: str
    purposes: frozenset[Purpose]
    granted_at: float
    expires_at: float | None = None
    revoked_at: float | None = None
    consent_id: str = ""
    """Stable identifier recorded on every resulting event, so an audit can
    point at the exact consent that authorised a specific identification."""

    camera_scope: frozenset[str] = field(default_factory=frozenset)
    """Empty means all cameras. When non-empty, consent covers only the listed
    camera ids — the mechanism a student uses to opt out of, say, the hostel
    corridor cameras while keeping the rest of campus."""

    def is_active(self, now: float | None = None) -> bool:
        t = now if now is not None else time.time()
        if self.revoked_at is not None and t >= self.revoked_at:
            return False
        if self.expires_at is not None and t >= self.expires_at:
            return False
        return True

    def covers_camera(self, camera_id: str) -> bool:
        return not self.camera_scope or camera_id in self.camera_scope


@dataclass(frozen=True, slots=True)
class ZonePolicy:
    """Per-zone policy. Resolved from a camera's configured zone.

    Zones exist because purpose limitation is rarely uniform. A biometric
    terminal in a library has a different necessity and a different consent
    expectation than a corridor camera, even at the same university.
    """

    zone: str
    allowed_purposes: frozenset[Purpose]
    require_consent: bool = True
    retention_days: int = 30
    anonymise_after_minutes: int | None = None
    """Minutes after which the event is stripped of student_id. Set on
    security cameras where the operational need is "was the corridor
    occupied" and the identity is not needed at all after the fact."""

    notes: str = ""


@dataclass(slots=True)
class ConsentRegistry:
    """In-memory consent store with a write-through audit log.

    Deliberately a synchronous, blocking check on the identity hot path. A
    cache with a TTL would mean a revoked consent still authorising
    identifications for the TTL duration, which fails s.12 in spirit and
    would not survive a review.
    """

    _records: dict[str, ConsentRecord] = field(default_factory=dict)
    _zones: dict[str, ZonePolicy] = field(default_factory=dict)
    _revocations: list[tuple[str, float]] = field(default_factory=list)

    def default_zone_policy(self) -> ZonePolicy:
        return ZonePolicy(
            zone="default",
            allowed_purposes=frozenset({Purpose.ATTENDANCE, Purpose.SAFETY}),
            require_consent=True,
            retention_days=30,
        )

    def set_zone_policy(self, policy: ZonePolicy) -> None:
        self._zones[policy.zone] = policy

    def zone_policy(self, zone: str) -> ZonePolicy:
        return self._zones.get(zone, self.default_zone_policy())

    def grant(self, record: ConsentRecord) -> None:
        self._records[record.student_id] = record

    def revoke(self, student_id: str, now: float | None = None) -> bool:
        """Revoke consent. Returns False if there was nothing to revoke."""
        t = now if now is not None else time.time()
        existing = self._records.get(student_id)
        if existing is None or not existing.is_active(t):
            return False
        self._records[student_id] = ConsentRecord(
            student_id=existing.student_id,
            purposes=frozenset(),
            granted_at=existing.granted_at,
            expires_at=existing.expires_at,
            revoked_at=t,
            consent_id=existing.consent_id,
            camera_scope=existing.camera_scope,
        )
        self._revocations.append((student_id, t))
        return True

    def check(
        self,
        student_id: str,
        purpose: Purpose,
        camera_id: str,
        zone: str,
        now: float | None = None,
    ) -> ConsentDecision:
        """Decide whether this identification is permitted. The gate."""
        t = now if now is not None else time.time()
        policy = self.zone_policy(zone)

        if purpose not in policy.allowed_purposes:
            return ConsentDecision(
                permitted=False,
                consent_id=None,
                purposes=frozenset(),
                reason=f"purpose {purpose.value} not permitted in zone {zone!r}",
            )

        if not policy.require_consent:
            return ConsentDecision(
                permitted=True,
                consent_id=None,
                purposes=frozenset({purpose}),
                reason="zone does not require consent",
            )

        record = self._records.get(student_id)
        if record is None:
            return ConsentDecision(
                permitted=False,
                consent_id=None,
                purposes=frozenset(),
                reason="no consent on record",
            )
        if not record.is_active(t):
            return ConsentDecision(
                permitted=False,
                consent_id=record.consent_id,
                purposes=frozenset(),
                reason="consent revoked or expired",
            )
        if purpose not in record.purposes:
            return ConsentDecision(
                permitted=False,
                consent_id=record.consent_id,
                purposes=record.purposes,
                reason=f"consent does not cover purpose {purpose.value}",
            )
        if not record.covers_camera(camera_id):
            return ConsentDecision(
                permitted=False,
                consent_id=record.consent_id,
                purposes=record.purposes,
                reason=f"consent excludes camera {camera_id}",
            )

        return ConsentDecision(
            permitted=True,
            consent_id=record.consent_id,
            purposes=record.purposes,
            reason="granted",
        )

    def retention_seconds(self, zone: str) -> int:
        return self.zone_policy(zone).retention_days * 86400

    def should_anonymise(self, zone: str, event_age_s: float) -> bool:
        policy = self.zone_policy(zone)
        if policy.anonymise_after_minutes is None:
            return False
        return event_age_s > policy.anonymise_after_minutes * 60

    def pending_erasures(self) -> list[tuple[str, float]]:
        return list(self._revocations)
