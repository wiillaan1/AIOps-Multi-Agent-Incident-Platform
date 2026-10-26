"""Load and validate versioned root-cause verification profiles."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.models.investigation import RootCauseVerificationProfile


class ProfileRegistry:
    """In-memory registry populated from validated YAML verification profiles."""

    def __init__(self, profile_dir: Path | None = None) -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        self.profile_dir = profile_dir or backend_dir / "data" / "root_cause_profiles"
        self._profiles: dict[str, RootCauseVerificationProfile] = {}

    def load(self) -> dict[str, RootCauseVerificationProfile]:
        """Load all YAML profiles, rejecting malformed, invalid, or duplicate entries."""
        profiles: dict[str, RootCauseVerificationProfile] = {}
        for profile_path in sorted(self.profile_dir.glob("*.yaml")):
            data = self._read_yaml(profile_path)
            try:
                profile = RootCauseVerificationProfile.model_validate(data)
            except ValidationError as exc:
                raise ValueError(f"invalid profile schema in {profile_path.name}: {exc}") from exc

            if profile.root_cause in profiles:
                raise ValueError(f"duplicate root_cause: {profile.root_cause}")
            profiles[profile.root_cause] = profile

        self._profiles = profiles
        return self._copy_profiles()

    def get(self, root_cause: str) -> RootCauseVerificationProfile | None:
        """Return a loaded profile by root-cause identifier, if present."""
        profile = self._profiles.get(root_cause)
        return profile.model_copy(deep=True) if profile is not None else None

    def list_profiles(self) -> list[RootCauseVerificationProfile]:
        """Return loaded profiles in deterministic root-cause order."""
        return [self._profiles[root_cause].model_copy(deep=True) for root_cause in sorted(self._profiles)]

    def validate_complete(self, required_causes: Iterable[str]) -> list[str]:
        """Return required root causes that have no successfully loaded profile."""
        return sorted({root_cause for root_cause in required_causes if root_cause not in self._profiles})

    def _copy_profiles(self) -> dict[str, RootCauseVerificationProfile]:
        return {
            root_cause: profile.model_copy(deep=True)
            for root_cause, profile in self._profiles.items()
        }

    @staticmethod
    def _read_yaml(profile_path: Path) -> dict[str, Any]:
        try:
            with profile_path.open(encoding="utf-8") as profile_file:
                data = yaml.safe_load(profile_file)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {profile_path.name}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"unable to read profile {profile_path.name}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"invalid profile schema in {profile_path.name}: expected a mapping")
        return data
