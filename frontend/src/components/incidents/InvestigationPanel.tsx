import { useEffect, useState, type ReactElement } from 'react';
import {
  FlaskConical,
  Loader2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  ShieldQuestion,
  Activity,
  ScrollText,
  Clock,
} from 'lucide-react';
import { useIncidents } from '@/hooks/useApi';
import type {
  AssertionOutcome,
  EvidenceRecord,
  InvestigationSnapshot,
  VerificationStatus,
} from '@/types';

interface InvestigationPanelProps {
  incidentId: string;
}

const STATE_LABELS: Record<string, string> = {
  pending: '待开始',
  investigating: '调查中',
  replanning: '重规划中',
  awaiting_execution: '等待执行回执',
  recovery_observing: '恢复观测中',
  recovered: '已恢复',
  heal_failed: '自愈失败',
  manual_escalation: '人工升级',
};

const VERIFICATION_META: Record<
  VerificationStatus,
  { label: string; color: string; icon: ReactElement }
> = {
  confirmed: { label: '已确认', color: 'text-emerald-400', icon: <CheckCircle2 className="w-4 h-4" /> },
  contradicted: { label: '已被反证', color: 'text-red-400', icon: <XCircle className="w-4 h-4" /> },
  inconclusive: { label: '证据不足', color: 'text-amber-400', icon: <HelpCircle className="w-4 h-4" /> },
  manual_only: { label: '仅人工', color: 'text-sky-400', icon: <ShieldQuestion className="w-4 h-4" /> },
};

function groupEvidence(evidence: EvidenceRecord[]) {
  const supporting = evidence.filter((e) => e.outcome === 'pass');
  const counter = evidence.filter((e) => e.outcome === 'counter' || e.outcome === 'fail');
  const unavailable = evidence.filter((e) => e.outcome === 'unavailable');
  return { supporting, counter, unavailable };
}

function sourceBadge(source: string): string {
  switch (source) {
    case 'prometheus':
      return 'bg-orange-400/10 text-orange-400';
    case 'loki':
      return 'bg-amber-400/10 text-amber-400';
    case 'health':
      return 'bg-emerald-400/10 text-emerald-400';
    case 'guardrail':
      return 'bg-rose-400/10 text-rose-400';
    default:
      return 'bg-slate-400/10 text-slate-400';
  }
}

function outcomeMeta(outcome: AssertionOutcome): { label: string; color: string } {
  switch (outcome) {
    case 'pass':
      return { label: '通过', color: 'text-emerald-400' };
    case 'counter':
      return { label: '反证', color: 'text-red-400' };
    case 'fail':
      return { label: '未达预期', color: 'text-amber-400' };
    case 'unavailable':
      return { label: '数据不可达', color: 'text-slate-400' };
    default:
      return { label: outcome, color: 'text-slate-400' };
  }
}

function EvidenceRow({ record }: { record: EvidenceRecord }): ReactElement {
  const outcome = outcomeMeta(record.outcome);
  const query = (record.query_summary?.query as string) ?? '—';
  return (
    <div className="bg-slate-900/40 rounded border border-slate-800 p-2.5 space-y-1">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${sourceBadge(record.source)}`}>
            {record.source}
          </span>
          <span className="text-xs font-mono text-foreground truncate">{record.assertion_id}</span>
        </div>
        <div className={`flex items-center gap-1 text-xs font-medium ${outcome.color} flex-shrink-0`}>
          {outcome.label}
          {record.score !== null && record.score !== undefined && (
            <span className="text-muted-foreground">×{record.score.toFixed(2)}</span>
          )}
        </div>
      </div>
      <div className="text-[11px] text-muted-foreground font-mono truncate">{query}</div>
      {record.observed_at && (
        <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
          <Clock className="w-3 h-3" />
          {new Date(record.observed_at).toLocaleString('zh-CN')}
        </div>
      )}
    </div>
  );
}

export function InvestigationPanel({ incidentId }: InvestigationPanelProps): ReactElement {
  const { getInvestigation, getInvestigationEvidence } = useIncidents();
  const [snapshot, setSnapshot] = useState<InvestigationSnapshot | null>(null);
  const [evidence, setEvidence] = useState<EvidenceRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const snap = (await getInvestigation(incidentId)) as InvestigationSnapshot | null;
        const ev = (await getInvestigationEvidence(incidentId)) as
          | { evidence?: EvidenceRecord[] }
          | null;
        if (cancelled) return;
        if (snap) setSnapshot(snap);
        setEvidence(ev?.evidence ?? []);
      } catch (e) {
        if (cancelled) return;
        const msg = (e as { detail?: string; message?: string })?.detail
          || (e as { message?: string })?.message
          || '获取调查快照失败';
        setError(msg);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [incidentId, getInvestigation, getInvestigationEvidence]);

  const { supporting, counter, unavailable } = groupEvidence(evidence);
  const latestHypothesis = snapshot?.hypotheses?.[snapshot.hypotheses.length - 1] ?? null;
  const audit = snapshot?.audit_summary ?? {};
  const dryRun = (audit.dry_run as Record<string, unknown> | undefined) ?? undefined;
  const nextStrategies = (audit.reflection_context as Record<string, unknown> | undefined)
    ?.next_strategies as string[] | undefined;
  const escalationReason = audit.manual_escalation_reason as string | undefined;
  const verification = snapshot
    ? VERIFICATION_META[snapshot.verification_status] ?? VERIFICATION_META.inconclusive
    : VERIFICATION_META.inconclusive;

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-sky-400" />
          <h3 className="text-base font-semibold text-foreground">RCA 调查验证</h3>
        </div>
        {snapshot && (
          <span className="text-xs px-2 py-0.5 bg-sky-400/10 text-sky-400 rounded">
            {STATE_LABELS[snapshot.state] ?? snapshot.state}
          </span>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
          <Loader2 className="w-4 h-4 animate-spin" />
          正在加载调查快照…
        </div>
      )}

      {error && !loading && (
        <div className="flex items-start gap-2 text-sm text-muted-foreground bg-slate-900/40 rounded p-3 border border-slate-800">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <div>{error}</div>
        </div>
      )}

      {snapshot && !loading && !error && (
        <div className="space-y-4">
          {/* Round + budget */}
          <div className="grid grid-cols-4 gap-2">
            <div className="bg-slate-900/40 rounded p-2 border border-slate-800">
              <div className="text-xs text-muted-foreground">RCA 轮次</div>
              <div className="text-sm font-semibold">
                {snapshot.rca_round}<span className="text-muted-foreground">/3</span>
              </div>
            </div>
            <div className="bg-slate-900/40 rounded p-2 border border-slate-800">
              <div className="text-xs text-muted-foreground">自愈尝试</div>
              <div className="text-sm font-semibold">
                {snapshot.heal_attempt}<span className="text-muted-foreground">/2</span>
              </div>
            </div>
            <div className="bg-slate-900/40 rounded p-2 border border-slate-800">
              <div className="text-xs text-muted-foreground">证据条数</div>
              <div className="text-sm font-semibold">{evidence.length}</div>
            </div>
            <div className="bg-slate-900/40 rounded p-2 border border-slate-800">
              <div className="text-xs text-muted-foreground">验证结论</div>
              <div className={`text-sm font-semibold flex items-center gap-1 ${verification.color}`}>
                {verification.icon}
                <span className="text-xs">{verification.label}</span>
              </div>
            </div>
          </div>

          {/* Hypothesis */}
          {latestHypothesis && (
            <div className="bg-slate-900/40 rounded p-3 border border-slate-800 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <Activity className="w-3.5 h-3.5" />
                  当前假设（第 {latestHypothesis.rca_round} 轮）
                </span>
                <span className="text-xs font-semibold text-foreground">
                  {(latestHypothesis.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="text-sm font-mono text-sky-300">{latestHypothesis.root_cause}</div>
              {latestHypothesis.profile_version && (
                <div className="text-[11px] text-muted-foreground">
                  Profile v{latestHypothesis.profile_version}
                </div>
              )}
            </div>
          )}

          {/* Dry-run plan (awaiting execution) */}
          {dryRun && (
            <div className="bg-slate-900/40 rounded p-3 border border-slate-700/60 space-y-1.5">
              <div className="text-xs text-muted-foreground flex items-center gap-1">
                <ScrollText className="w-3.5 h-3.5" />
                Dry-run 方案（仅模拟，未执行）
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-muted-foreground">Playbook: </span>
                  <span className="font-mono text-foreground">
                    {String(dryRun.playbook_id ?? '—')}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Target: </span>
                  <span className="font-mono text-foreground">
                    {String(dryRun.target_resource ?? '—')}
                  </span>
                </div>
              </div>
              <div className="text-[11px] text-amber-400/80">
                ⚠️ Dry-run 不等同于执行成功；恢复需等待可信执行回执后判定。
              </div>
            </div>
          )}

          {/* Evidence grouped */}
          <div className="space-y-2">
            <div className="text-xs text-muted-foreground">证据分组</div>
            <div>
              <div className="text-xs text-emerald-400 mb-1">支持证据（{supporting.length}）</div>
              {supporting.length === 0 ? (
                <div className="text-[11px] text-muted-foreground pl-2">暂无</div>
              ) : (
                <div className="space-y-1.5">{supporting.map((e) => <EvidenceRow key={e.evidence_id} record={e} />)}</div>
              )}
            </div>
            <div>
              <div className="text-xs text-red-400 mb-1">反证 / 未达预期（{counter.length}）</div>
              {counter.length === 0 ? (
                <div className="text-[11px] text-muted-foreground pl-2">暂无</div>
              ) : (
                <div className="space-y-1.5">{counter.map((e) => <EvidenceRow key={e.evidence_id} record={e} />)}</div>
              )}
            </div>
            <div>
              <div className="text-xs text-slate-400 mb-1">数据不可达（{unavailable.length}）</div>
              {unavailable.length === 0 ? (
                <div className="text-[11px] text-muted-foreground pl-2">暂无</div>
              ) : (
                <div className="space-y-1.5">{unavailable.map((e) => <EvidenceRow key={e.evidence_id} record={e} />)}</div>
              )}
            </div>
          </div>

          {/* Next strategies */}
          {nextStrategies && nextStrategies.length > 0 && (
            <div className="bg-amber-900/20 border border-amber-700/40 rounded p-2.5">
              <div className="text-xs text-amber-400 mb-1">下一轮策略</div>
              <div className="flex flex-wrap gap-1.5">
                {nextStrategies.map((s) => (
                  <span key={s} className="px-2 py-0.5 bg-amber-400/10 text-amber-400 rounded text-xs">
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Escalation */}
          {escalationReason && (
            <div className="bg-rose-900/20 border border-rose-700/40 rounded p-2.5">
              <div className="text-xs text-rose-400 flex items-center gap-1">
                <AlertCircle className="w-3.5 h-3.5" />
                人工升级
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">{escalationReason}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
