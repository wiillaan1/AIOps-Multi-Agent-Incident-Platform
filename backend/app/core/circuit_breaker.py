"""
Circuit Breaker - 熔断器

经典三态熔断器（CLOSED / OPEN / HALF_OPEN），用于保护 LLM 等外部服务调用：
- CLOSED：正常调用，记录失败次数
- OPEN：熔断打开，所有调用直接失败，等待 timeout 后进入 HALF_OPEN
- HALF_OPEN：放行少量探测请求，成功则恢复 CLOSED，失败则重新 OPEN

参考：Michael Nygard, "Release It!" 第 4 章 + Microsoft Azure Architecture Center。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar


class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开探测


class CircuitBreakerOpen(Exception):
    """熔断器打开时调用抛出的异常"""
    def __init__(self, breaker_name: str, state: CircuitState):
        self.breaker_name = breaker_name
        self.state = state
        super().__init__(f"[{breaker_name}] 熔断器 {state.value}，调用被拦截")


T = TypeVar("T")


@dataclass
class CircuitBreakerStats:
    """熔断器统计（用于可观测性）"""
    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    total_failures: int
    total_short_circuits: int
    last_failure_time: float | None
    last_state_change_time: float


class CircuitBreaker:
    """
    线程安全的熔断器。

    用法：
        breaker = CircuitBreaker("llm", failure_threshold=3, timeout=60)
        try:
            result = breaker.call(some_function, arg1, arg2)
        except CircuitBreakerOpen:
            # 走 fallback
            ...
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        half_open_max_calls: int = 1,
        expected_exceptions: tuple[type[BaseException], ...] = (Exception,),
    ):
        """
        参数：
            name: 熔断器名称（用于日志/指标）
            failure_threshold: 连续失败多少次触发熔断（CLOSED → OPEN）
            success_threshold: HALF_OPEN 连续成功多少次后恢复（HALF_OPEN → CLOSED）
            timeout_seconds: OPEN 状态持续多久后转 HALF_OPEN
            half_open_max_calls: HALF_OPEN 同时允许多少个探测调用
            expected_exceptions: 哪些异常算"失败"。默认所有 Exception。
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        self.expected_exceptions = expected_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_in_flight = 0
        self._last_failure_time: float | None = None
        self._last_state_change_time = time.time()

        self._total_calls = 0
        self._total_failures = 0
        self._total_short_circuits = 0

        self._lock = threading.RLock()

    # ---------- 公开属性 ----------
    @property
    def state(self) -> CircuitState:
        with self._lock:
            # 懒检查 OPEN → HALF_OPEN 转换（避免后台线程）
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def stats(self) -> CircuitBreakerStats:
        with self._lock:
            return CircuitBreakerStats(
                name=self.name,
                state=self.state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                total_calls=self._total_calls,
                total_failures=self._total_failures,
                total_short_circuits=self._total_short_circuits,
                last_failure_time=self._last_failure_time,
                last_state_change_time=self._last_state_change_time,
            )

    # ---------- 核心调用 ----------
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        通过熔断器调用 func。

        抛出：
            CircuitBreakerOpen: 熔断器 OPEN 时
            func 的异常: 调用失败时
        """
        self._before_call()
        try:
            result = func(*args, **kwargs)
        except self.expected_exceptions:
            self._on_failure()
            raise
        self._on_success()
        return result

    def reset(self) -> None:
        """手动重置熔断器到 CLOSED"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_in_flight = 0
            self._last_state_change_time = time.time()

    # ---------- 内部 ----------
    def _should_attempt_reset(self) -> bool:
        return (
            self._last_failure_time is not None
            and time.time() - self._last_failure_time >= self.timeout_seconds
        )

    def _before_call(self) -> None:
        with self._lock:
            self._total_calls += 1

            # 懒检查 OPEN → HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._total_short_circuits += 1
                    raise CircuitBreakerOpen(self.name, self._state)

            # HALF_OPEN 限制 in-flight 探测
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_in_flight >= self.half_open_max_calls:
                    self._total_short_circuits += 1
                    raise CircuitBreakerOpen(self.name, self._state)
                self._half_open_in_flight += 1

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # 渐进恢复：成功一次减一次失败计数（防毛刺）
                self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self) -> None:
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        if self._state != new_state:
            old = self._state
            self._state = new_state
            self._last_state_change_time = time.time()
            # 重置 state-specific 计数
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
                self._half_open_in_flight = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._success_count = 0
                self._half_open_in_flight = 0
            elif new_state == CircuitState.OPEN:
                self._success_count = 0
            # 状态变化不打日志（避免循环依赖），调用方读 stats() 即可
