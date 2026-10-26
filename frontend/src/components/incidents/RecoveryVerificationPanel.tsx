import { useEffect, useState, type ReactElement } from 'react';
import {
  HeartPulse,
  Loader2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Clock,
  RotateCcw,
  ShieldAlert,
} from 'lucide-react';
import { useIncidents } from '@/hooks/useApi';
import type { InvestigationSnapshot, RecoveryOutcome, HealExecutionReceipt } from '@/types';

interface RecoveryVerificationPanelProps {
  incidentId: string;
}

const OUTCOME_META: Record<
  RecoveryOutcome,
  { label: string; color: string; icon: ReactElement }
> = {
  recovered: { label: '已恢复', color: 'text-emerald-400', icon: <CheckCircle2 className="w-4 h-4" /> },
  degraded: { label: '部分恢复', color: 'text-amber-400', icon: <AlertCircle className="w-4 h-4" /> },
  failed: { label: '未恢复', color: 'text-red-400', icon: <XCircle className="w-4 h-4" /> },
  inconclusive: { label: '证据不足', color: 'text-slate-400', icon: <Clock className="w-4 h-4" /> },
};

function receiptStatusBadge(status: HealExecutionReceipt['status']): string {
  switch (status) {
    case 'succeeded':
      return 'bg-emerald-400/10 text-emerald-400';
    case 'failed':
      return 'bg-red-400/10 text-red-400';
    case 'rejected':
      return 'bg-slate-400/10 text-slate-400';
    default:
      return 'bg-slate-400/10 text-slate-400';
  }
}

export function RecoveryVerificationPanel({ incidentId }: RecoveryVerificationPanelProps): ReactElement {
  const { getInvestigation, requestRecoveryVerification, loading: submitting } = useIncidents();
  const [snapshot, setSnapshot] = useState<InvestigationSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionMsg, setActionMsg] = useState<string | null>(null);

  const reload = async () => {
    setLoading(true);
    setError(null);
    try {
      const snap = (await getInvestigation(incidentId)) as InvestigationSnapshot | null;
      setSnapshot(snap);
    } catch (e) {
      const msg = (e as { detail?: string; message?: string })?.detail
        || (e as { message?: string })?.message
        || '获取恢复验证失败';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      if (cancelled) return;
      await reload();
    };
    load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incidentId, getInvestigation]);

  const handleVerify = async () => {
    setActionMsg(null);
    setError(null);
    try {
      await requestRecoveryVerification(incidentId);
      await reload();
      setActionMsg('已请求稳定窗口恢复验证');
    } catch (e) {
      const msg = (e as { detail?: string; message?: string })?.detail
        || (e as { message?: string })?.message
        || '恢复验证请求失败';
      setError(msg);
    }
  };

  const receipts = snapshot?.receipts ?? [];
  const latestReceipt = receipts[receipts.length - 1] ?? null;
  const latestVerification = snapshot?.recovery_verifications?.[
    snapshot.recovery_verifications.length - 1
  ] ?? null;
  const audit = snapshot?.audit_summary ?? {};
  const planB = audit.plan_b_suggestion as Record<string, unknown> | undefined;
  const isAwaiting = snapshot?.state === 'awaiting_execution';
  const isObserving = snapshot?.state === 'recovery_observing';

  const outcome = latestVerification
    ? OUTCOME_META[latestVerification.outcome] ?? OUTCOME_META.inconclusive
    : null;

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <HeartPulse className="w-5 h-5 text-emerald-400" />
          <h3 className="text-base font-semibold text-foreground">恢复验证</h3>
        </div>
        {snapshot && (
          <button
            type="button"
            onClick={reload}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            刷新
          </button>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
          <Loader2 className="w-4 h-4 animate-spin" />
          正在加载恢复验证…
        </div>
      )}

      {error && !loading && (
        <div className="flex items-start gap-2 text-sm text-muted-foreground bg-slate-900/40 rounded p-3 border border-slate-800">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <div>{error}</div>
        </div>
      )}

      {snapshot && !loading && (
        <div className="space-y-4">
          {/* State banner */}
          {isAwaiting && (
            <div className="bg-amber-900/20 border border-amber-700/40 rounded p-2.5 text-xs text-amber-300">
              ⚠️ 当前为 Dry-run 方案，等待外部可信执行回执；恢复尚未判定。
            </div>
          )}
          {isObserving && (
            <div className="bg-sky-900/20 border border-sky-700/40 rounded p-2.5 text-xs text-sky-300">
              稳定窗口观测中，等待连续采样满足要求后判定恢复。
            </div>
          )}

          {/* Receipt status */}
          <div>
            <div className="text-xs text-muted-foreground mb-1">执行回执</div>
            {latestReceipt ? (
              <div className="bg-slate-900/40 rounded border border-slate-800 p-2.5 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-foreground">
                    {latestReceipt.playbook_id}
                  </span>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${receiptStatusBadge(latestReceipt.status)}`}>
                    {latestReceipt.status}
                  </span>
                </div>
                <div className="text-[11px] text-muted-foreground">
                  目标 {latestReceipt.target_resource} · 幂等键 {latestReceipt.idempotency_key}
                </div>
                {latestReceipt.completed_at && (
                  <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    完成于 {new Date(latestReceipt.completed_at).toLocaleString('zh-CN')}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-[11px] text-muted-foreground">
                尚无执行回执（Dry-run 不计入）
              </div>
            )}
          </div>

          {/* Recovery outcome + signals */}
          {latestVerification && outcome && (
            <div className="bg-slate-900/40 rounded border border-slate-800 p-2.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">恢复判定</span>
                <div className={`flex items-center gap-1 text-sm font-semibold ${outcome.color}`}>
                  {outcome.icon}
                  {outcome.label}
                </div>
              </div>
              {latestVerification.unavailable_sources.length > 0 && (
                <div className="text-[11px] text-slate-400">
                  不可达信号源: {latestVerification.unavailable_sources.join(', ')}
                </div>
              )}
              {latestVerification.signals.length > 0 && (
                <div className="space-y-1">
                  {latestVerification.signals.map((sig, idx) => (
                    <div key={idx} className="flex items-center justify-between text-[11px]">
                      <span className="text-muted-foreground">{String(sig.name ?? `signal-${idx}`)}</span>
                      <span className={
                        sig.status === 'pass' ? 'text-emerald-400'
                          : sig.status === 'fail' ? 'text-red-400'
                            : 'text-slate-400'
                      }>
                        {String(sig.status ?? '—')}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Plan B / rollback recommendation */}
          {planB && (
            <div className="bg-orange-900/20 border border-orange-700/40 rounded p-2.5 space-y-1">
              <div className="text-xs text-orange-400 flex items-center gap-1">
                <RotateCcw className="w-3.5 h-3.5" />
                Plan B 建议（仅建议，不自动执行）
              </div>
              <div className="text-xs font-mono text-foreground">
                {String(planB.playbook_id ?? planB.id ?? '—')}
              </div>
              {planB.playbook_name ? (
                <div className="text-[11px] text-muted-foreground">{String(planB.playbook_name)}</div>
              ) : null}
            </div>
          )}

          {/* Escalation */}
          {snapshot.state === 'manual_escalation' && (
            <div className="bg-rose-900/20 border border-rose-700/40 rounded p-2.5 space-y-1">
              <div className="text-xs text-rose-400 flex items-center gap-1">
                <ShieldAlert className="w-3.5 h-3.5" />
                已升级人工
              </div>
              <div className="text-xs text-muted-foreground">
                {String(audit.manual_escalation_reason ?? '已达调查或恢复上限')}
              </div>
            </div>
          )}

          {/* Action */}
          {actionMsg && (
            <div className="text-[11px] text-emerald-400">{actionMsg}</div>
          )}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleVerify}
              disabled={submitting || !latestReceipt || latestReceipt.status !== 'succeeded'}
              className="text-xs px-3 py-1.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-700/40 hover:bg-emerald-500/20 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              请求恢复验证
            </button>
          </div>
          {!latestReceipt && (
            <div className="text-[11px] text-muted-foreground">
              提示：恢复验证需先有可信 succeeded 执行回执；Dry-run 不会触发恢复判定。
            </div>
          )}
        </div>
      )}
    </div>
  );
}
