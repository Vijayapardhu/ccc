"""Persistence: presence events, gallery vectors, consent records."""

from __future__ import annotations

from campus.store.db import Database, build_presence_from_commit

__all__ = ["Database", "build_presence_from_commit"]
