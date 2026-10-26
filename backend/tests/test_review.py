"""
Review 模块测试

覆盖：
1. Review 模型（avg_score / is_passing）
2. ReviewStore CRUD（submit/get/list_recent/count）
3. ReviewStore.list_pending_incidents
4. ReviewStore.stats
5. CLI submit/list/stats（用 subprocess 调）
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.review.models import (
    DEFAULT_DIMENSIONS,
    Review,
    ReviewDecision,
    ReviewDimension,
)
from app.review.store import ReviewStore, get_review_store, reset_review_store


@pytest.fixture
def store(tmp_path) -> ReviewStore:
    db_path = tmp_path / "reviews.db"
    reset_review_store()
    s = ReviewStore(str(db_path))
    yield s
    reset_review_store()


# =====================================================================
# 模型测试
# =====================================================================

class TestReviewModel:
    def test_avg_score(self):
        r = Review(scores={"accuracy": 4, "explainability": 5, "overall": 4})
        assert r.avg_score() == pytest.approx(4.33, 0.01)

    def test_avg_score_empty(self):
        r = Review()
        assert r.avg_score() == 0.0

    def test_is_passing_accept(self):
        r = Review(decision="accept", scores={"overall": 3})
        assert r.is_passing() is True

    def test_is_passing_reject(self):
        r = Review(decision="reject", scores={"overall": 5})
        assert r.is_passing() is False

    def test_is_passing_low_overall(self):
        r = Review(decision="accept", scores={"overall": 2})
        assert r.is_passing() is False

    def test_default_dimensions_count(self):
        assert len(DEFAULT_DIMENSIONS) == 4


# =====================================================================
# ReviewStore 测试
# =====================================================================

class TestReviewStore:
    def test_submit_returns_id(self, store: ReviewStore):
        r = Review(
            incident_id="inc-001",
            reviewer="zhangsan",
            decision="accept",
            scores={"accuracy": 4, "overall": 4},
        )
        saved = store.submit(r)
        assert saved.id is not None
        assert isinstance(saved.id, int)

    def test_get(self, store: ReviewStore):
        r = Review(
            incident_id="inc-002", reviewer="lisi",
            decision="accept", scores={"overall": 5},
        )
        saved = store.submit(r)
        fetched = store.get(saved.id)
        assert fetched is not None
        assert fetched.incident_id == "inc-002"
        assert fetched.reviewer == "lisi"
        assert fetched.scores["overall"] == 5

    def test_list_recent(self, store: ReviewStore):
        for i in range(3):
            store.submit(Review(
                incident_id=f"inc-{i}", reviewer="x",
                decision="accept", scores={"overall": 3},
            ))
        reviews = store.list_recent(limit=10)
        assert len(reviews) == 3

    def test_count(self, store: ReviewStore):
        assert store.count() == 0
        store.submit(Review(incident_id="i1", reviewer="x", decision="accept", scores={"overall": 3}))
        assert store.count() == 1

    def test_list_pending_incidents(self, store: ReviewStore):
        # 评 2 个
        store.submit(Review(incident_id="i1", reviewer="x", decision="accept", scores={"overall": 3}))
        store.submit(Review(incident_id="i2", reviewer="x", decision="accept", scores={"overall": 3}))
        # 给 4 个查
        pending = store.list_pending_incidents(["i1", "i2", "i3", "i4"])
        assert pending == ["i3", "i4"]

    def test_list_pending_empty_input(self, store: ReviewStore):
        assert store.list_pending_incidents([]) == []


# =====================================================================
# 统计测试
# =====================================================================

class TestReviewStoreStats:
    def test_stats_empty(self, store: ReviewStore):
        s = store.stats()
        assert s["total"] == 0
        assert s["avg_overall"] == 0.0

    def test_stats_with_data(self, store: ReviewStore):
        store.submit(Review(incident_id="i1", reviewer="zhangsan",
                            decision="accept", scores={"overall": 4}))
        store.submit(Review(incident_id="i2", reviewer="zhangsan",
                            decision="accept", scores={"overall": 5}))
        store.submit(Review(incident_id="i3", reviewer="lisi",
                            decision="reject", scores={"overall": 2}))

        s = store.stats()
        assert s["total"] == 3
        assert s["by_decision"]["accept"] == 2
        assert s["by_decision"]["reject"] == 1
        assert s["by_reviewer"]["zhangsan"] == 2
        assert s["by_reviewer"]["lisi"] == 1
        assert s["avg_overall"] == pytest.approx(3.67, 0.01)
        # pass_rate = accept / (accept+reject) = 2/3
        assert s["pass_rate"] == pytest.approx(0.667, 0.01)

    def test_stats_pass_rate_only_accept_reject(self, store: ReviewStore):
        # 只有 needs_improvement 不计 pass_rate
        store.submit(Review(incident_id="i1", reviewer="x",
                            decision="needs_improvement", scores={"overall": 4}))
        s = store.stats()
        assert s["pass_rate"] == 0.0  # 没有 accept/reject


# =====================================================================
# CLI 测试（subprocess）
# =====================================================================

class TestReviewCLI:
    @pytest.fixture
    def isolated_db(self, tmp_path, monkeypatch):
        """每个测试用独立 db"""
        db_path = tmp_path / "reviews.db"
        monkeypatch.setenv("REVIEW_DB_PATH", str(db_path))
        return db_path

    def test_cli_submit_and_list(self, isolated_db):
        # 提交
        result = subprocess.run(
            [sys.executable, "-m", "app.review", "submit",
             "--incident", "inc-001",
             "--reviewer", "test_user",
             "--decision", "accept",
             "--accuracy", "4",
             "--explainability", "4",
             "--actionability", "5",
             "--overall", "4",
             "--comments", "合理"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0, f"stdout: {result.stdout}, stderr: {result.stderr}"
        assert "已提交" in result.stdout

        # list
        result = subprocess.run(
            [sys.executable, "-m", "app.review", "list"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0
        assert "inc-001" in result.stdout
        assert "test_user" in result.stdout

    def test_cli_stats(self, isolated_db):
        # 先提交一条
        subprocess.run(
            [sys.executable, "-m", "app.review", "submit",
             "--incident", "inc-001", "--reviewer", "u1",
             "--decision", "accept",
             "--accuracy", "5", "--explainability", "5",
             "--actionability", "5", "--overall", "5"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.review", "stats"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0
        assert "总评分数: 1" in result.stdout
        assert "accept" in result.stdout

    def test_cli_missing_required(self, isolated_db):
        result = subprocess.run(
            [sys.executable, "-m", "app.review", "submit",
             "--incident", "inc-001"],  # 缺 reviewer/decision/scores
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode != 0

    def test_cli_invalid_decision(self, isolated_db):
        result = subprocess.run(
            [sys.executable, "-m", "app.review", "submit",
             "--incident", "i1", "--reviewer", "u1",
             "--decision", "INVALID_DECISION",
             "--accuracy", "4", "--explainability", "4",
             "--actionability", "4", "--overall", "4"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode != 0
        assert "必须是" in result.stdout

    def test_cli_score_out_of_range(self, isolated_db):
        result = subprocess.run(
            [sys.executable, "-m", "app.review", "submit",
             "--incident", "i1", "--reviewer", "u1",
             "--decision", "accept",
             "--accuracy", "10",  # 超出 1-5
             "--explainability", "4",
             "--actionability", "4", "--overall", "4"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode != 0
        assert "1-5" in result.stdout

    def test_cli_list_pending(self, isolated_db):
        # 评 1 个
        subprocess.run(
            [sys.executable, "-m", "app.review", "submit",
             "--incident", "i1", "--reviewer", "u1",
             "--decision", "accept",
             "--accuracy", "4", "--explainability", "4",
             "--actionability", "4", "--overall", "4"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        # 查 3 个
        result = subprocess.run(
            [sys.executable, "-m", "app.review", "list-pending", "i1", "i2", "i3"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0
        assert "i2" in result.stdout
        assert "i3" in result.stdout
        assert "i1" not in result.stdout.split("=== ")[1]  # i1 已评分

    def test_cli_show(self, isolated_db):
        # 提交
        subprocess.run(
            [sys.executable, "-m", "app.review", "submit",
             "--incident", "inc-show", "--reviewer", "u1",
             "--decision", "accept",
             "--accuracy", "4", "--explainability", "4",
             "--actionability", "4", "--overall", "4",
             "--comments", "test comment"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        # show
        result = subprocess.run(
            [sys.executable, "-m", "app.review", "show", "inc-show"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0
        assert "u1" in result.stdout
        assert "test comment" in result.stdout
