"""
数据集适配器 —— 把真实数据集转换为 BadCaseEntry 格式

[v3 E2 改造] 2026-09-01

支持的数据集：
- aiops2020: 清华 AIOps Challenge 2020（多维根因定位 / 故障诊断）
- loghub_hdfs: Loghub HDFS_v1（分布式系统日志异常）
- smd: Server Machine Dataset (OmniAnomaly 5周服务器指标)

每个适配器：
1. 从公开数据集提取真实场景
2. 转换为 BadCaseEntry 格式
3. 写入 v1_dev.json / v1_test.json / v1_holdout.json（三分离）

设计原则：
- 真实数据集溯源（source_ref 指向原始 case）
- 适配器可重跑（idempotent）
- 字段映射清晰（alert / evidence / expected_root_cause）
"""
