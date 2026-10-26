"""v3 E8: Microsoft AIOpsLab adapter —— 把 87 个 problem 转成 Badcase

数据源：https://github.com/microsoft/AIOpsLab/blob/main/aiopslab/orchestrator/problems/registry.py
Leaderboard：https://microsoft.github.io/AIOpsLab/pages/leaderboard

设计：
- 静态生成 87 个 Badcase 草稿（每个含 schema + 推断的 expected_root_cause）
- 不真接入 AIOpsLab 框架（沙箱跑不动 docker）
- 提供 reference leaderboard scores 作为对照基线
- 学生本地有 docker 时，可直接对接 AIOpsLab registry.py 拿真评测

输出：
- v1_dev.json / v1_test.json / v1_holdout.json（追加 AIOpsLab badcase）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 87 个 AIOpsLab problems（从 microsoft/AIOpsLab registry.py 提取）
# 每条: (problem_id, application, fault_type, task_type, difficulty)
AIOPSLAB_PROBLEMS = [
    # K8s target port misconfig (3 services × 4 tasks)
    ("k8s_target_port-misconfig-detection-1", "user-service", "k8s_target_port-misconfig", "detection", "easy"),
    ("k8s_target_port-misconfig-localization-1", "user-service", "k8s_target_port-misconfig", "localization", "medium"),
    ("k8s_target_port-misconfig-analysis-1", "user-service", "k8s_target_port-misconfig", "analysis", "hard"),
    ("k8s_target_port-misconfig-mitigation-1", "user-service", "k8s_target_port-misconfig", "mitigation", "hard"),
    ("k8s_target_port-misconfig-detection-2", "text-service", "k8s_target_port-misconfig", "detection", "easy"),
    ("k8s_target_port-misconfig-localization-2", "text-service", "k8s_target_port-misconfig", "localization", "medium"),
    ("k8s_target_port-misconfig-analysis-2", "text-service", "k8s_target_port-misconfig", "analysis", "hard"),
    ("k8s_target_port-misconfig-mitigation-2", "text-service", "k8s_target_port-misconfig", "mitigation", "hard"),
    ("k8s_target_port-misconfig-detection-3", "post-storage-service", "k8s_target_port-misconfig", "detection", "easy"),
    ("k8s_target_port-misconfig-localization-3", "post-storage-service", "k8s_target_port-misconfig", "localization", "medium"),
    ("k8s_target_port-misconfig-analysis-3", "post-storage-service", "k8s_target_port-misconfig", "analysis", "hard"),
    ("k8s_target_port-misconfig-mitigation-3", "post-storage-service", "k8s_target_port-misconfig", "mitigation", "hard"),

    # MongoDB auth missing
    ("auth_miss_mongodb-detection-1", "mongodb", "auth_miss_mongodb", "detection", "easy"),
    ("auth_miss_mongodb-localization-1", "mongodb", "auth_miss_mongodb", "localization", "medium"),
    ("auth_miss_mongodb-analysis-1", "mongodb", "auth_miss_mongodb", "analysis", "medium"),
    ("auth_miss_mongodb-mitigation-1", "mongodb", "auth_miss_mongodb", "mitigation", "hard"),

    # MongoDB auth revoke (2 services × 4 tasks)
    ("revoke_auth_mongodb-detection-1", "mongodb-geo", "revoke_auth_mongodb", "detection", "easy"),
    ("revoke_auth_mongodb-localization-1", "mongodb-geo", "revoke_auth_mongodb", "localization", "medium"),
    ("revoke_auth_mongodb-analysis-1", "mongodb-geo", "revoke_auth_mongodb", "analysis", "medium"),
    ("revoke_auth_mongodb-mitigation-1", "mongodb-geo", "revoke_auth_mongodb", "mitigation", "hard"),
    ("revoke_auth_mongodb-detection-2", "mongodb-rate", "revoke_auth_mongodb", "detection", "easy"),
    ("revoke_auth_mongodb-localization-2", "mongodb-rate", "revoke_auth_mongodb", "localization", "medium"),
    ("revoke_auth_mongodb-analysis-2", "mongodb-rate", "revoke_auth_mongodb", "analysis", "medium"),
    ("revoke_auth_mongodb-mitigation-2", "mongodb-rate", "revoke_auth_mongodb", "mitigation", "hard"),

    # MongoDB user unregistered
    ("user_unregistered_mongodb-detection-1", "mongodb-geo", "user_unregistered_mongodb", "detection", "easy"),
    ("user_unregistered_mongodb-localization-1", "mongodb-geo", "user_unregistered_mongodb", "localization", "medium"),
    ("user_unregistered_mongodb-analysis-1", "mongodb-geo", "user_unregistered_mongodb", "analysis", "medium"),
    ("user_unregistered_mongodb-mitigation-1", "mongodb-geo", "user_unregistered_mongodb", "mitigation", "hard"),
    ("user_unregistered_mongodb-detection-2", "mongodb-rate", "user_unregistered_mongodb", "detection", "easy"),
    ("user_unregistered_mongodb-localization-2", "mongodb-rate", "user_unregistered_mongodb", "localization", "medium"),
    ("user_unregistered_mongodb-analysis-2", "mongodb-rate", "user_unregistered_mongodb", "analysis", "medium"),
    ("user_unregistered_mongodb-mitigation-2", "mongodb-rate", "user_unregistered_mongodb", "mitigation", "hard"),

    # Hotel reservation misconfig
    ("misconfig_app_hotel_res-detection-1", "hotel_res", "misconfig_app_hotel_res", "detection", "easy"),
    ("misconfig_app_hotel_res-localization-1", "hotel_res", "misconfig_app_hotel_res", "localization", "medium"),
    ("misconfig_app_hotel_res-analysis-1", "hotel_res", "misconfig_app_hotel_res", "analysis", "medium"),
    ("misconfig_app_hotel_res-mitigation-1", "hotel_res", "misconfig_app_hotel_res", "mitigation", "hard"),

    # Social network scale zero
    ("scale_pod_zero_social_net-detection-1", "social_net", "scale_pod_zero_social_net", "detection", "easy"),
    ("scale_pod_zero_social_net-localization-1", "social_net", "scale_pod_zero_social_net", "localization", "medium"),
    ("scale_pod_zero_social_net-analysis-1", "social_net", "scale_pod_zero_social_net", "analysis", "medium"),
    ("scale_pod_zero_social_net-mitigation-1", "social_net", "scale_pod_zero_social_net", "mitigation", "hard"),

    # Assign pod to non-existent node
    ("assign_to_non_existent_node_social_net-detection-1", "social_net", "assign_to_non_existent_node_social_net", "detection", "easy"),
    ("assign_to_non_existent_node_social_net-localization-1", "social_net", "assign_to_non_existent_node_social_net", "localization", "medium"),
    ("assign_to_non_existent_node_social_net-analysis-1", "social_net", "assign_to_non_existent_node_social_net", "analysis", "medium"),
    ("assign_to_non_existent_node_social_net-mitigation-1", "social_net", "assign_to_non_existent_node_social_net", "mitigation", "hard"),

    # Container kill
    ("container_kill-detection", "default", "container_kill", "detection", "easy"),
    ("container_kill-localization", "default", "container_kill", "localization", "medium"),

    # Pod failure / kill / network issues (hotel_res)
    ("pod_failure_hotel_res-detection-1", "hotel_res", "pod_failure_hotel_res", "detection", "easy"),
    ("pod_failure_hotel_res-localization-1", "hotel_res", "pod_failure_hotel_res", "localization", "medium"),
    ("pod_kill_hotel_res-detection-1", "hotel_res", "pod_kill_hotel_res", "detection", "easy"),
    ("pod_kill_hotel_res-localization-1", "hotel_res", "pod_kill_hotel_res", "localization", "medium"),
    ("network_loss_hotel_res-detection-1", "hotel_res", "network_loss_hotel_res", "detection", "easy"),
    ("network_loss_hotel_res-localization-1", "hotel_res", "network_loss_hotel_res", "localization", "medium"),
    ("network_delay_hotel_res-detection-1", "hotel_res", "network_delay_hotel_res", "detection", "easy"),
    ("network_delay_hotel_res-localization-1", "hotel_res", "network_delay_hotel_res", "localization", "medium"),

    # Noop detection
    ("noop_detection_hotel_reservation-1", "hotel_reservation", "noop_detection", "detection", "easy"),
    ("noop_detection_social_network-1", "social_network", "noop_detection", "detection", "easy"),
    ("noop_detection_astronomy_shop-1", "astronomy_shop", "noop_detection", "detection", "easy"),

    # Astronomy shop various faults
    ("astronomy_shop_ad_service_failure-detection-1", "astronomy_shop_ad_service", "astronomy_shop_ad_service_failure", "detection", "easy"),
    ("astronomy_shop_ad_service_failure-localization-1", "astronomy_shop_ad_service", "astronomy_shop_ad_service_failure", "localization", "medium"),
    ("astronomy_shop_ad_service_high_cpu-detection-1", "astronomy_shop_ad_service", "astronomy_shop_ad_service_high_cpu", "detection", "easy"),
    ("astronomy_shop_ad_service_high_cpu-localization-1", "astronomy_shop_ad_service", "astronomy_shop_ad_service_high_cpu", "localization", "medium"),
    ("astronomy_shop_ad_service_manual_gc-detection-1", "astronomy_shop_ad_service", "astronomy_shop_ad_service_manual_gc", "detection", "easy"),
    ("astronomy_shop_ad_service_manual_gc-localization-1", "astronomy_shop_ad_service", "astronomy_shop_ad_service_manual_gc", "localization", "medium"),
    ("astronomy_shop_cart_service_failure-detection-1", "astronomy_shop_cart_service", "astronomy_shop_cart_service_failure", "detection", "easy"),
    ("astronomy_shop_cart_service_failure-localization-1", "astronomy_shop_cart_service", "astronomy_shop_cart_service_failure", "localization", "medium"),
    ("astronomy_shop_image_slow_load-detection-1", "astronomy_shop_image", "astronomy_shop_image_slow_load", "detection", "easy"),
    ("astronomy_shop_image_slow_load-localization-1", "astronomy_shop_image", "astronomy_shop_image_slow_load", "localization", "medium"),
    ("astronomy_shop_kafka_queue_problems-detection-1", "astronomy_shop_kafka", "astronomy_shop_kafka_queue_problems", "detection", "easy"),
    ("astronomy_shop_kafka_queue_problems-localization-1", "astronomy_shop_kafka", "astronomy_shop_kafka_queue_problems", "localization", "medium"),
    ("astronomy_shop_kafka_queue_problems-mitigation-1", "astronomy_shop_kafka", "astronomy_shop_kafka_queue_problems", "mitigation", "hard"),
    ("astronomy_shop_loadgenerator_flood_homepage-detection-1", "astronomy_shop_loadgenerator", "astronomy_shop_loadgenerator_flood_homepage", "detection", "easy"),
    ("astronomy_shop_loadgenerator_flood_homepage-localization-1", "astronomy_shop_loadgenerator", "astronomy_shop_loadgenerator_flood_homepage", "localization", "medium"),
    ("astronomy_shop_payment_service_failure-detection-1", "astronomy_shop_payment_service", "astronomy_shop_payment_service_failure", "detection", "easy"),
    ("astronomy_shop_payment_service_failure-localization-1", "astronomy_shop_payment_service", "astronomy_shop_payment_service_failure", "localization", "medium"),
    ("astronomy_shop_payment_service_unreachable-detection-1", "astronomy_shop_payment_service", "astronomy_shop_payment_service_unreachable", "detection", "easy"),
    ("astronomy_shop_payment_service_unreachable-localization-1", "astronomy_shop_payment_service", "astronomy_shop_payment_service_unreachable", "localization", "medium"),
    ("astronomy_shop_product_catalog_service_failure-detection-1", "astronomy_shop_product_catalog", "astronomy_shop_product_catalog_service_failure", "detection", "easy"),
    ("astronomy_shop_product_catalog_service_failure-localization-1", "astronomy_shop_product_catalog", "astronomy_shop_product_catalog_service_failure", "localization", "medium"),
    ("astronomy_shop_recommendation_service_cache_failure-detection-1", "astronomy_shop_recommendation_service", "astronomy_shop_recommendation_service_cache_failure", "detection", "easy"),
    ("astronomy_shop_recommendation_service_cache_failure-localization-1", "astronomy_shop_recommendation_service", "astronomy_shop_recommendation_service_cache_failure", "localization", "medium"),

    # Redeploy without PV
    ("redeploy_without_PV-detection-1", "default", "redeploy_without_PV", "detection", "easy"),
    ("redeploy_without_PV-analysis-1", "default", "redeploy_without_PV", "analysis", "medium"),
    ("redeploy_without_PV-mitigation-1", "default", "redeploy_without_PV", "mitigation", "hard"),

    # Wrong bin usage
    ("wrong_bin_usage-detection-1", "default", "wrong_bin_usage", "detection", "easy"),
    ("wrong_bin_usage-localization-1", "default", "wrong_bin_usage", "localization", "medium"),
    ("wrong_bin_usage-analysis-1", "default", "wrong_bin_usage", "analysis", "medium"),
    ("wrong_bin_usage-mitigation-1", "default", "wrong_bin_usage", "mitigation", "hard"),

    # Flower (federated learning)
    ("flower_node_stop-detection", "flower", "flower_node_stop", "detection", "easy"),
    ("flower_model_misconfig-detection", "flower", "flower_model_misconfig", "detection", "easy"),
]


# AIOpsLab Leaderboard 参考分数（来源：microsoft.github.io/AIOpsLab/pages/leaderboard，2026-08）
AIOPSLAB_LEADERBOARD = {
    "OpsAgent (GPT-4)": {"average": 78.75, "detection": 100, "localization": 60, "diagnosis": 83.34, "mitigation": 66.67, "org": "AIOpsLab & NKU"},
    "FLASH (GPT-4)": {"average": 59.27, "detection": 100, "localization": 46.15, "diagnosis": 36.36, "mitigation": 54.55, "org": "AIOpsLab"},
    "REACT (GPT-4)": {"average": 53.15, "detection": 76.92, "localization": 53.85, "diagnosis": 45.45, "mitigation": 36.36, "org": "AIOpsLab"},
    "DeepSeek-R1": {"average": 50.47, "detection": 78.57, "localization": 58.33, "diagnosis": 35.0, "mitigation": 30.0, "org": "THU"},
    "GPT-4 w Shell": {"average": 49.74, "detection": 69.23, "localization": 61.54, "diagnosis": 40.9, "mitigation": 27.27, "org": "AIOpsLab"},
    "FLASH (Llama3-8b)": {"average": 33.34, "detection": 80, "localization": 20, "diagnosis": 0, "mitigation": 33.34, "org": "THU"},
    "GPT-3.5 w Shell": {"average": 15.73, "detection": 23.07, "localization": 30.77, "diagnosis": 9.09, "mitigation": 0, "org": "AIOpsLab"},
}


# Difficulty → confidence_min 映射
DIFFICULTY_TO_CONF = {
    "easy": 0.7,
    "medium": 0.8,
    "hard": 0.9,
}

# Difficulty → expected_category 映射
DIFFICULTY_TO_CATEGORY = {
    "easy": "single_metric",        # detection 类，单一信号
    "medium": "compound",           # localization / analysis 类，多信号融合
    "hard": "regression_guard",     # mitigation 类，需要强约束
}


def aiopslab_problem_to_badcase(problem: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """AIOpsLab problem 转 Badcase dict"""
    problem_id, application, fault_type, task_type, difficulty = problem

    # 构造 alert / evidence（按 AIOpsLab schema 风格）
    alert = {
        "service": application,
        "metric_name": "task_success_rate",
        "value": 0.0,
        "task_type": task_type,
        "application": application,
        "fault_type": fault_type,
    }
    evidence = {
        "task_type": task_type,
        "application": application,
        "fault_type": fault_type,
        "difficulty": difficulty,
        "aiopslab_problem_id": problem_id,
    }

    # expected_root_cause：基于 fault_type 推断
    expected_root_cause = f"{fault_type}_in_{application}"

    # expected_confidence_min：按 difficulty
    expected_confidence_min = DIFFICULTY_TO_CONF[difficulty]

    return {
        "id": f"bc-aiopslab-{problem_id}",
        "category": DIFFICULTY_TO_CATEGORY[difficulty],
        "type": fault_type,
        "alert": alert,
        "evidence": evidence,
        "expected_root_cause": expected_root_cause,
        "expected_confidence_min": expected_confidence_min,
        "tags": ["aiopslab", task_type, difficulty, fault_type],
        "badcase_class": "aiopslab_benchmark",
        "identified_flaw": f"AIOpsLab problem {problem_id}: {fault_type} in {application} ({task_type})",
        "keywords_for_retrieval": [task_type, fault_type, application, difficulty],
        "severity": "high" if difficulty == "hard" else ("medium" if difficulty == "medium" else "low"),
        "eval_split": "dev",  # 后面 split_to_three 会重分
        "source": "aiopslab",
        "description": (
            f"AIOpsLab benchmark problem: {task_type} of {fault_type} in {application}. "
            f"Difficulty: {difficulty}."
        ),
    }


def main():
    """把所有 AIOpsLab problem 转 Badcase 并按 70/20/10 切分"""
    # 使用 common 工具
    from data.adapters.common import deterministic_split, merge_or_replace_split

    print(f"Converting {len(AIOPSLAB_PROBLEMS)} AIOpsLab problems to Badcase...")
    badcases = [aiopslab_problem_to_badcase(p) for p in AIOPSLAB_PROBLEMS]

    # 按 70/20/10 切分
    dev, test, holdout = deterministic_split(badcases)

    # 写入（追加模式，不覆盖其他 source）
    dev_path, dev_added = merge_or_replace_split(dev, "dev", "aiopslab")
    test_path, test_added = merge_or_replace_split(test, "test", "aiopslab")
    holdout_path, holdout_added = merge_or_replace_split(holdout, "holdout", "aiopslab")

    print(f"\n=== AIOpsLab adapter summary ===")
    print(f"Total: {len(badcases)}")
    print(f"dev: {len(dev)} (newly added: {len(dev_added)})")
    print(f"test: {len(test)} (newly added: {len(test_added)})")
    print(f"holdout: {len(holdout)} (newly added: {len(holdout_added)})")
    print(f"By task_type:")
    by_task = {}
    for bc in badcases:
        tt = bc["alert"]["task_type"]
        by_task.setdefault(tt, 0)
        by_task[tt] += 1
    for tt, cnt in sorted(by_task.items()):
        print(f"  {tt}: {cnt}")
    print(f"\nLeaderboard reference (for comparison):")
    for name, scores in sorted(AIOPSLAB_LEADERBOARD.items(), key=lambda x: -x[1]["average"])[:5]:
        print(f"  {name}: avg={scores['average']:.2f} (det={scores['detection']:.0f}, loc={scores['localization']:.0f}, dia={scores['diagnosis']:.0f}, mit={scores['mitigation']:.0f})")


if __name__ == "__main__":
    main()