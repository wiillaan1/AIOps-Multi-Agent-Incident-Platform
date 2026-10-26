"""
AIOps Agent Platform - Configuration Management

使用 pydantic-settings 管理所有配置，支持从环境变量和 .env 文件加载。
所有配置项均以 AIOPS_ 为前缀的环境变量可覆盖。
"""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 支持的 LLM provider（白名单，避免任意字符串注入）
LLMProvider = Literal["openai", "anthropic", "ollama", "mimimax", "deepseek"]


class LLMConfig(BaseSettings):
    """LLM 模型配置"""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: LLMProvider = Field(default="openai", description="LLM 提供商")
    api_key: str = Field(default="", description="API 密钥")
    base_url: str = Field(default="", description="OpenAI 兼容 API 地址（为空则用 provider 默认）")
    model: str = Field(default="gpt-4o-mini", description="模型名称")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="采样温度")
    max_tokens: int = Field(default=4096, gt=0, description="最大生成 token 数")
    timeout_seconds: int = Field(default=60, gt=0, description="请求超时(秒)")
    max_retries: int = Field(default=3, ge=0, description="最大重试次数")

    # ----- 成本核算（审查报告七、中期补 #7）-----
    # 价格以 USD/1k tokens 为单位。默认 0 表示不计费（仅记 token 数）。
    # 实际值由用户在 .env 配置（如 glm-5.3、deepseek、各类 provider 计费标准）。
    input_cost_per_1k: float = Field(
        default=0.0, ge=0.0, description="输入 token 单价（USD/1k tokens）"
    )
    output_cost_per_1k: float = Field(
        default=0.0, ge=0.0, description="输出 token 单价（USD/1k tokens）"
    )
    cost_currency: str = Field(
        default="USD", description="成本计价货币（当前仅展示用）"
    )

    @model_validator(mode="after")
    def _check_provider_base_url(self) -> "LLMConfig":
        """provider 特定校验：mimimax/ollama 必须配置 base_url"""
        if self.provider in ("mimimax", "ollama") and not self.base_url:
            raise ValueError(
                f"LLM provider={self.provider} 必须设置 LLM_BASE_URL "
                f"（mimimax: https://.../v1；ollama: http://localhost:11434）"
            )
        return self

    def is_key_present(self) -> bool:
        """检查 API key 是否已配置（非空）"""
        return bool(self.api_key and self.api_key.strip())

    def warn_if_key_missing(self) -> None:
        """启动时调用：API key 缺失时不阻断，仅警告（允许规则引擎模式跑）"""
        if not self.is_key_present():
            warnings.warn(
                "[LLMConfig] LLM_API_KEY 未配置！系统将以规则引擎模式运行。"
                "LLM 调用会降级到 fallback。生产环境必须设置。",
                stacklevel=2,
            )


class LangfuseConfig(BaseSettings):
    """Langfuse 可观测性配置"""

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    public_key: str = Field(default="", description="Langfuse Public Key")
    secret_key: str = Field(default="", description="Langfuse Secret Key")
    host: str = Field(default="https://cloud.langfuse.com", description="Langfuse 服务地址")
    enabled: bool = Field(default=False, description="是否启用 Langfuse")
    release: str = Field(default="1.0.0", description="应用版本号")
    environment: str = Field(default="development", description="运行环境")


class DatabaseConfig(BaseSettings):
    """数据库配置"""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = Field(
        default="sqlite:///./data/aiops.db",
        description="数据库连接 URL",
    )
    pool_size: int = Field(default=10, gt=0, description="连接池大小")
    max_overflow: int = Field(default=20, ge=0, description="最大溢出连接数")
    echo: bool = Field(default=False, description="是否打印 SQL 语句")


class AgentConfig(BaseSettings):
    """Agent 行为配置"""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    execution_timeout_seconds: int = Field(
        default=120, gt=0, description="Agent 执行超时(秒)"
    )
    max_iterations: int = Field(
        default=10, gt=0, description="最大迭代次数"
    )
    parallel_workers: int = Field(
        default=3, gt=0, description="并行工作线程数"
    )
    heal_dry_run: bool = Field(
        default=True, description="自愈操作是否仅模拟"
    )
    rca_min_confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="根因分析最低置信度"
    )
    # PPT 思路主链路 RCA 合成优化（rca_synthesis_v2.py）。
    # 默认关闭：开启后修正相似度语义/冲突降确定性/rejected 消费/provisional 记忆门禁。
    # 关闭=主链路 _synthesize_analysis 字节不变。兼容 env AIOPS_USE_RCA_V2。
    rca_v2_enabled: bool = Field(
        default=False,
        description="是否启用 PPT 思路主链路 RCA 合成优化（相似度修正/冲突降确定性/rejected/provisional）",
    )
    # RCA 调查验证闭环（InvestigationLoopEngine）配置。
    # 默认关闭：主路径仍走旧管道；开启后由引擎驱动候选验证/反证/重规划/Dry-run。
    verification_enabled: bool = Field(
        default=False, description="是否启用调查验证闭环（引擎驱动）"
    )
    verification_max_rca_rounds: int = Field(
        default=3, ge=1, le=3, description="调查验证最大 RCA 轮次"
    )
    verification_max_heal_attempts: int = Field(
        default=2, ge=1, le=2, description="调查验证最大自愈尝试次数（仅 Dry-run）"
    )
    verification_query_timeout_seconds: int = Field(
        default=10, ge=1, le=60, description="单条 Profile 查询超时(秒)"
    )
    verification_query_max_rows: int = Field(
        default=100, ge=1, le=10_000, description="单条查询返回最大条数"
    )
    verification_settle_seconds: int = Field(
        default=30, ge=0, le=3600, description="执行回执后稳定窗口前静默秒数"
    )
    verification_observation_seconds: int = Field(
        default=120, ge=1, le=3600, description="恢复观测窗口秒数"
    )


class MemoryConfig(BaseSettings):
    """记忆系统配置"""

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    short_term_max_items: int = Field(
        default=100, gt=0, description="短期记忆最大条目数"
    )
    long_term_max_items: int = Field(
        default=10000, gt=0, description="长期记忆最大条目数"
    )
    similarity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="记忆相似度阈值"
    )
    retention_days: int = Field(
        default=90, gt=0, description="记忆保留天数"
    )


class WebSocketConfig(BaseSettings):
    """WebSocket 配置"""

    model_config = SettingsConfigDict(
        env_prefix="WS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    heartbeat_interval_seconds: int = Field(
        default=30, gt=0, description="心跳间隔(秒)"
    )
    max_connections: int = Field(
        default=100, gt=0, description="最大连接数"
    )


class AppConfig(BaseSettings):
    """应用主配置 - 聚合所有子配置"""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "testing", "staging", "production"] = Field(
        default="development", description="运行环境"
    )
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, gt=0, description="监听端口")
    log_level: str = Field(default="INFO", description="日志级别")
    debug: bool = Field(default=False, description="调试模式")

    # CORS 配置
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="CORS 允许的源",
    )
    api_key_header: str = Field(
        default="X-API-Key", description="API Key 请求头名称"
    )

    # 子配置实例
    llm: LLMConfig = Field(default_factory=LLMConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: str | list) -> list[str]:
        """从逗号分隔字符串解析 CORS 源列表"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.env == "development"

    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.env == "production"


# 全局配置单例
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """
    获取全局配置实例（懒加载单例模式）

    Returns:
        AppConfig: 应用配置对象
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    """
    强制重新加载配置（用于配置热更新）

    Returns:
        AppConfig: 重新加载的应用配置对象
    """
    global _config
    _config = AppConfig()
    return _config


def validate_secrets(config: AppConfig | None = None) -> dict:
    """
    启动时校验 secrets 配置（不阻断，仅报告）

    Returns:
        dict: 校验报告 {"critical": [...], "warnings": [...], "info": [...]}

    设计原则：
    - critical: 必须修复才能跑生产（缺失会立即报错）
    - warnings: 当前能跑但有风险（demo 允许）
    - info: 仅信息提示
    """
    cfg = config or get_config()
    report = {"critical": [], "warnings": [], "info": []}

    # LLM API key（warning，不阻断——demo 允许无 key 跑规则模式）
    if not cfg.llm.is_key_present():
        report["warnings"].append(
            "LLM_API_KEY 未配置：系统将以规则引擎模式运行，"
            "LLM 调用会降级到 fallback。生产环境必须设置。"
        )
    else:
        report["info"].append(f"LLM provider={cfg.llm.provider} model={cfg.llm.model} key 已配置")

    # provider base_url（critical for mimimax/ollama）
    if cfg.llm.provider in ("mimimax", "ollama") and not cfg.llm.base_url:
        report["critical"].append(
            f"LLM provider={cfg.llm.provider} 必须配置 LLM_BASE_URL"
        )

    # Langfuse（warning——启用时才需要）
    if cfg.langfuse.enabled:
        if not cfg.langfuse.public_key or not cfg.langfuse.secret_key:
            report["warnings"].append(
                "LANGFUSE_ENABLED=true 但 public_key/secret_key 未配置，"
                "Langfuse trace 会被禁用"
            )
        else:
            report["info"].append(f"Langfuse 已启用: host={cfg.langfuse.host}")

    # 生产环境额外检查
    if cfg.is_production:
        if cfg.app_config_debug if hasattr(cfg, "app_config_debug") else cfg.debug:
            report["critical"].append("生产环境不能开启 APP_DEBUG=true")
        if cfg.llm.api_key and len(cfg.llm.api_key) < 20:
            report["warnings"].append("LLM_API_KEY 长度过短，可能不是有效 key")

    return report
