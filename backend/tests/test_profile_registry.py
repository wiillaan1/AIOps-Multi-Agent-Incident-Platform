from pathlib import Path

import pytest

from app.models.investigation import StandardRootCause
from app.services.profile_registry import ProfileRegistry


def valid_profile(root_cause: str = "dependency_failure") -> str:
    return f'''root_cause: {root_cause}
version: "1.0"
manual_only: false
assertions:
  - id: direct-evidence
    source: prometheus
    query_template: 'rate(errors{{service="{{{{ service }}}}", target="{{{{ target }}}}"}}[5m])'
    evidence_level: direct
    weight: 1.0
    expectation:
      operator: greater_than
      threshold: 0.1
counter_evidence:
  - id: healthy-target
    source: health
    query_template: 'health{{service="{{{{ service }}}}", target="{{{{ target }}}}"}}'
    weight: 1.0
    expectation:
      operator: contains
      threshold: healthy
next_strategies:
  inconclusive: [expand_time_window]
  contradicted: [test_recent_deployment]
decision:
  minimum_score: 0.65
  require_direct_evidence: true
  maximum_counter_evidence: 0.3
recovery:
  settle_seconds: 0
  observation_seconds: 1
  consecutive_samples: 1
'''


def test_registry_loads_valid_versioned_yaml_profile(tmp_path: Path):
    (tmp_path / "dependency_failure.yaml").write_text(valid_profile(), encoding="utf-8")

    profiles = ProfileRegistry(tmp_path).load()

    assert profiles["dependency_failure"].version == "1.0"


def test_registry_get_and_list_profiles_after_load(tmp_path: Path):
    (tmp_path / "dependency_failure.yaml").write_text(valid_profile(), encoding="utf-8")
    registry = ProfileRegistry(tmp_path)
    registry.load()

    assert registry.get("dependency_failure").root_cause == "dependency_failure"
    assert [profile.root_cause for profile in registry.list_profiles()] == ["dependency_failure"]
    assert registry.get("missing") is None


def test_registry_reports_missing_required_cause(tmp_path: Path):
    registry = ProfileRegistry(tmp_path)
    registry.load()

    assert registry.validate_complete(["dependency_failure"]) == ["dependency_failure"]


def test_registry_rejects_duplicate_root_cause(tmp_path: Path):
    (tmp_path / "first.yaml").write_text(valid_profile(), encoding="utf-8")
    (tmp_path / "second.yaml").write_text(valid_profile(), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate root_cause"):
        ProfileRegistry(tmp_path).load()


def test_registry_rejects_malformed_yaml(tmp_path: Path):
    (tmp_path / "broken.yaml").write_text("root_cause: [not valid", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YAML"):
        ProfileRegistry(tmp_path).load()


def test_registry_rejects_invalid_profile_schema(tmp_path: Path):
    (tmp_path / "invalid.yaml").write_text("root_cause: dependency_failure\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid profile schema"):
        ProfileRegistry(tmp_path).load()


def test_default_profiles_cover_every_standard_root_cause():
    registry = ProfileRegistry()
    registry.load()

    assert registry.validate_complete(StandardRootCause.__args__) == []
    assert {profile.root_cause for profile in registry.list_profiles()} == set(StandardRootCause.__args__)


@pytest.mark.parametrize(
    "root_cause",
    [root_cause for root_cause in StandardRootCause.__args__ if root_cause != "unknown"],
)
def test_automatic_default_profiles_have_direct_counter_and_scoped_evidence(root_cause: str):
    profile = ProfileRegistry().load()[root_cause]
    rules = [*profile.assertions, *profile.counter_evidence]

    assert profile.manual_only is False
    assert any(assertion.evidence_level == "direct" for assertion in profile.assertions)
    assert profile.counter_evidence
    assert all("{{ service }}" in rule.query_template for rule in rules)
    assert all("{{ target }}" in rule.query_template for rule in rules)


def test_registry_returns_deep_copies_that_cannot_mutate_cached_profile(tmp_path: Path):
    (tmp_path / "dependency_failure.yaml").write_text(valid_profile(), encoding="utf-8")
    registry = ProfileRegistry(tmp_path)
    loaded = registry.load()

    loaded["dependency_failure"].assertions[0].weight = 0.0
    fetched = registry.get("dependency_failure")
    listed = registry.list_profiles()

    assert fetched is not None
    assert fetched.assertions[0].weight == 1.0
    assert listed[0].assertions[0].weight == 1.0
