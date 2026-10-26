"""
Tests for docker-compose / Loki configuration files.

These tests don't require docker, they only validate YAML structure and
that the wiring matches what the runtime expects.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/ -> aiops-agent-platform/


def _load_yaml(rel_path: str) -> dict:
    with open(REPO_ROOT / rel_path) as fh:
        return yaml.safe_load(fh)


# ============================ docker-compose.yml (prod) ============================


def test_prod_compose_has_backend_service():
    compose = _load_yaml("docker-compose.yml")
    assert "backend" in compose["services"]


def test_prod_compose_has_loki_service():
    compose = _load_yaml("docker-compose.yml")
    assert "loki" in compose["services"]
    loki = compose["services"]["loki"]
    assert "grafana/loki" in loki["image"]
    assert "3100:3100" in loki["ports"]
    assert loki["restart"] == "unless-stopped"


def test_prod_compose_has_promtail_service():
    compose = _load_yaml("docker-compose.yml")
    assert "promtail" in compose["services"]
    promtail = compose["services"]["promtail"]
    assert "grafana/promtail" in promtail["image"]
    assert "/var/run/docker.sock:/var/run/docker.sock:ro" in promtail["volumes"]
    assert "depends_on" in promtail


def test_prod_compose_backend_has_loki_url():
    compose = _load_yaml("docker-compose.yml")
    backend_env = compose["services"]["backend"]["environment"]
    assert "LOKI_URL=http://loki:3100" in backend_env


def test_prod_compose_has_loki_data_volume():
    compose = _load_yaml("docker-compose.yml")
    assert "loki-data" in compose["volumes"]


def test_prod_compose_loki_health_check():
    compose = _load_yaml("docker-compose.yml")
    loki = compose["services"]["loki"]
    hc = loki["healthcheck"]
    # test 字段是 list[str]（CMD + 命令 + 参数），/ready 应出现在 URL 字符串里
    joined = " ".join(hc["test"])
    assert "/ready" in joined


# ============================ docker-compose.dev.yml ============================


def test_dev_compose_has_loki_service():
    compose = _load_yaml("docker-compose.dev.yml")
    assert "loki" in compose["services"]


def test_dev_compose_has_promtail_service():
    compose = _load_yaml("docker-compose.dev.yml")
    assert "promtail" in compose["services"]


def test_dev_compose_backend_has_loki_url():
    compose = _load_yaml("docker-compose.dev.yml")
    backend_env = compose["services"]["backend-dev"]["environment"]
    assert "LOKI_URL=http://loki:3100" in backend_env


# ============================ Loki configs ============================


def test_loki_prod_config_has_30day_retention():
    cfg = _load_yaml("infrastructure/loki/loki-prod-config.yaml")
    assert cfg["limits_config"]["retention_period"] == "720h"


def test_loki_dev_config_has_7day_retention():
    cfg = _load_yaml("infrastructure/loki/loki-config.yaml")
    assert cfg["limits_config"]["retention_period"] == "168h"


def test_loki_prod_config_has_compactor():
    cfg = _load_yaml("infrastructure/loki/loki-prod-config.yaml")
    assert "compactor" in cfg
    assert cfg["compactor"]["retention_enabled"] is True


def test_loki_prod_config_documents_s3_migration():
    """生产配置必须包含 S3 迁移示例（注释形式可读）。"""
    with open(REPO_ROOT / "infrastructure/loki/loki-prod-config.yaml") as fh:
        text = fh.read()
    assert "s3" in text.lower()
    assert "对象存储切换示例" in text or "storage" in text


# ============================ Promtail configs ============================


def test_promtail_prod_config_targets_docker_sd():
    cfg = _load_yaml("infrastructure/promtail/promtail-prod-config.yaml")
    jobs = cfg["scrape_configs"]
    docker_jobs = [j for j in jobs if j["job_name"] == "docker"]
    assert docker_jobs, "promtail prod must include docker sd"
    assert "/var/run/docker.sock" in docker_jobs[0]["docker_sd_configs"][0]["host"]


def test_promtail_prod_has_aiops_label_filter():
    """生产 Promtail 应当只采集打了 aiops=true 标签的容器日志。"""
    cfg = _load_yaml("infrastructure/promtail/promtail-prod-config.yaml")
    docker_job = next(
        j for j in cfg["scrape_configs"] if j["job_name"] == "docker"
    )
    filters = docker_job["docker_sd_configs"][0].get("filters", [])
    label_filters = [f for f in filters if f.get("name") == "label"]
    assert any("aiops=true" in (f.get("values") or []) for f in label_filters)