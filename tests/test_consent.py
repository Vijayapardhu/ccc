"""DPDP consent enforcement."""

from __future__ import annotations

import pytest

from campus.consent.registry import (
    ConsentRecord,
    ConsentRegistry,
    Purpose,
    ZonePolicy,
)


def record(**kw) -> ConsentRecord:
    base = {
        "student_id": "S1001",
        "purposes": frozenset({Purpose.ATTENDANCE, Purpose.SAFETY}),
        "granted_at": 1000.0,
        "consent_id": "consent-abc",
    }
    return ConsentRecord(**{**base, **kw})


@pytest.fixture
def registry() -> ConsentRegistry:
    r = ConsentRegistry()
    r.set_zone_policy(
        ZonePolicy("default", frozenset({Purpose.ATTENDANCE, Purpose.SAFETY}))
    )
    r.set_zone_policy(
        ZonePolicy("corridor", frozenset({Purpose.SAFETY}), retention_days=7)
    )
    return r


class TestGrantAndCheck:
    def test_no_consent_means_no_identification(self, registry):
        d = registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "default", now=1000.0)
        assert not d.permitted
        assert d.reason == "no consent on record"

    def test_granted_consent_permits(self, registry):
        registry.grant(record())
        d = registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "default", now=1000.0)
        assert d.permitted
        assert d.consent_id == "consent-abc"

    def test_purpose_limitation(self, registry):
        """Consent for attendance does not cover research. This is the
        requirement that gets botched most often, so it is a closed enum."""
        registry.set_zone_policy(
            ZonePolicy("default", frozenset({Purpose.ATTENDANCE, Purpose.RESEARCH}))
        )
        registry.grant(record(purposes=frozenset({Purpose.ATTENDANCE})))
        assert registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "default", now=1000.0).permitted
        denied = registry.check("S1001", Purpose.RESEARCH, "cam-1", "default", now=1000.0)
        assert not denied.permitted
        assert "does not cover purpose" in denied.reason

    def test_zone_policy_is_checked_before_consent(self, registry):
        """A zone that does not permit the purpose at all is denied regardless
        of what the student consented to — the zone is the outer bound."""
        registry.grant(record(purposes=frozenset({Purpose.RESEARCH})))
        denied = registry.check("S1001", Purpose.RESEARCH, "cam-1", "default", now=1000.0)
        assert not denied.permitted
        assert "not permitted in zone" in denied.reason

    def test_zone_restricts_purposes(self, registry):
        """A corridor camera is a safety camera. Attendance collection from it
        is not authorised regardless of what the student consented to."""
        registry.grant(record())
        d = registry.check("S1001", Purpose.ATTENDANCE, "cor-1", "corridor", now=1000.0)
        assert not d.permitted
        assert "not permitted in zone" in d.reason

    def test_zone_allows_its_permitted_purpose(self, registry):
        registry.grant(record())
        assert registry.check("S1001", Purpose.SAFETY, "cor-1", "corridor", now=1000.0).permitted

    def test_camera_scope_opt_out(self, registry):
        """The mechanism a student uses to opt out of specific cameras while
        keeping the rest of campus."""
        registry.grant(record(camera_scope=frozenset({"cam-1", "cam-2"})))
        assert registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "default", now=1000.0).permitted
        denied = registry.check("S1001", Purpose.ATTENDANCE, "hostel-3", "default", now=1000.0)
        assert not denied.permitted
        assert "excludes camera" in denied.reason

    def test_zone_without_consent_requirement(self, registry):
        """A non-biometric zone (a bicycle rack, say) can identify without
        consent because no individual is being identified."""
        registry.set_zone_policy(
            ZonePolicy("public", frozenset({Purpose.ATTENDANCE}), require_consent=False)
        )
        d = registry.check("S1001", Purpose.ATTENDANCE, "cam-9", "public", now=1000.0)
        assert d.permitted
        assert "does not require consent" in d.reason

    def test_unknown_zone_falls_back_to_default(self, registry):
        registry.grant(record())
        assert registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "unmapped", now=1000.0).permitted


class TestRevocation:
    def test_revocation_takes_effect_immediately(self, registry):
        """Not cached with a TTL: a revoked consent authorising identifications
        for the TTL duration fails the Act in substance."""
        registry.grant(record())
        assert registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "default", now=1000.0).permitted
        assert registry.revoke("S1001", now=1001.0)
        d = registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "default", now=1001.0)
        assert not d.permitted
        assert d.reason == "consent revoked or expired"

    def test_revoking_twice_reports_false(self, registry):
        registry.grant(record())
        assert registry.revoke("S1001", now=1000.0)
        assert not registry.revoke("S1001", now=1001.0)

    def test_revoking_nothing_reports_false(self, registry):
        assert not registry.revoke("S9999", now=1000.0)

    def test_revocation_queues_erasure(self, registry):
        """The erasure worker needs to know who to purge."""
        registry.grant(record())
        registry.revoke("S1001", now=1000.0)
        assert ("S1001", 1000.0) in registry.pending_erasures()

    def test_expiry(self, registry):
        registry.grant(record(expires_at=2000.0))
        assert registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "default", now=1999.0).permitted
        assert not registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "default", now=2001.0).permitted


class TestRetention:
    def test_retention_follows_the_zone(self, registry):
        assert registry.retention_seconds("default") == 30 * 86400
        assert registry.retention_seconds("corridor") == 7 * 86400

    def test_anonymisation_window(self, registry):
        registry.set_zone_policy(
            ZonePolicy("corridor", frozenset({Purpose.SAFETY}),
                       anonymise_after_minutes=60)
        )
        assert not registry.should_anonymise("corridor", 30 * 60)
        assert registry.should_anonymise("corridor", 90 * 60)

    def test_no_anonymisation_when_unset(self, registry):
        assert not registry.should_anonymise("default", 10_000)


class TestPurposeEnum:
    def test_purposes_are_a_closed_set(self):
        """Adding a purpose is a code change, which forces someone to think
        about the consent text, rather than a string that defaults to allowed."""
        assert {p.value for p in Purpose} == {
            "attendance", "safety", "access_control",
            "space_utilisation", "research",
        }

    def test_consent_still_usable_after_grant(self, registry):
        registry.grant(record())
        d = registry.check("S1001", Purpose.ATTENDANCE, "cam-1", "default", now=1000.0)
        assert d.allows(Purpose.ATTENDANCE)
        assert not d.allows(Purpose.RESEARCH)
