import { useEffect, useState, type ReactElement } from 'react';
import { Link } from 'react-router-dom';
import {
  Search,
  BookOpen,
  Sparkles,
  Loader2,
  AlertCircle,
  Database,
  Filter,
  ArrowRight,
  CheckCircle2,
} from 'lucide-react';
import { useRunbookSearch } from '@/hooks/useApi';

interface SearchHit {
  doc_id: string;
  incident_id: string;
  version_number: number;
  status: string;
  service: string;
  root_cause: string;
  confidence: number;
  title: string;
  change_note: string;
  published_by: string;
  created_at: string;
  score: number | null;
  snippet: string;
}

function highlight(text: string, query: string): ReactElement {
  if (!query.trim()) return <>{text}</>;
  const q = query.trim();
  const lower = text.toLowerCase();
  const idx = lower.indexOf(q.toLowerCase());
  if (idx < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, idx)}
      <mark className="bg-amber-400/30 text-amber-100 px-0.5 rounded">
        {text.slice(idx, idx + q.length)}
      </mark>
      {text.slice(idx + q.length)}
    </>
  );
}

function StatusPill({ status }: { status: string }) {
  const map: Record<string, { bg: string; text: string }> = {
    published: { bg: 'bg-emerald-400/10', text: 'text-emerald-400' },
    archived: { bg: 'bg-slate-400/10', text: 'text-slate-400' },
    draft: { bg: 'bg-amber-400/10', text: 'text-amber-400' },
  };
  const style = map[status] ?? map.draft;
  return (
    <span
      className={`px-2 py-0.5 rounded text-xs ${style.bg} ${style.text}`}
    >
      {status}
    </span>
  );
}

function ScoreBar({ score }: { score: number | null }) {
  if (score === null || score === undefined) {
    return <span className="text-xs text-muted-foreground">n/a</span>;
  }
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${
            pct >= 70
              ? 'bg-emerald-400'
              : pct >= 40
                ? 'bg-amber-400'
                : 'bg-slate-400'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-muted-foreground w-8">{pct}%</span>
    </div>
  );
}

export function RunbookSearchPage() {
  const { searchRunbooks, getRunbookStats, data, loading, error } =
    useRunbookSearch();

  const [query, setQuery] = useState('');
  const [service, setService] = useState('');
  const [minConfidence, setMinConfidence] = useState('');
  const [stats, setStats] = useState<{ available: boolean; count: number } | null>(
    null
  );
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [searched, setSearched] = useState(false);

  // 拉索引统计（首次进入）
  useEffect(() => {
    let cancelled = false;
    const fetchStats = async () => {
      try {
        const result = (await getRunbookStats()) as
          | { available: boolean; count: number }
          | null;
        if (!cancelled && result) setStats(result);
      } catch {
        if (!cancelled) setStats({ available: false, count: 0 });
      }
    };
    fetchStats();
    return () => {
      cancelled = true;
    };
  }, [getRunbookStats]);

  // 当 data 更新（hook 内部 state），同步到本地 hits
  useEffect(() => {
    if (data?.hits) {
      setHits(data.hits as unknown as SearchHit[]);
      setSearched(true);
    }
  }, [data]);

  const runSearch = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim()) return;
    setSearched(false);
    await searchRunbooks({
      q: query.trim(),
      top_k: 10,
      service: service.trim() || undefined,
      min_confidence: minConfidence ? Number(minConfidence) : undefined,
    });
    // data 变化后上面的 useEffect 会更新 hits
  };

  return (
    <div className="space-y-6">
      {/* 顶部搜索区 */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-1">
          <BookOpen className="w-5 h-5 text-purple-400" />
          <h2 className="text-base font-semibold text-foreground">
            Runbook 检索
          </h2>
          <span className="text-xs text-muted-foreground">
            语义检索已发布的 Runbook 快照
          </span>
        </div>

        {stats && (
          <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
            <span className="flex items-center gap-1">
              <Database className="w-3 h-3" />
              索引{' '}
              {stats.available ? (
                <span className="text-emerald-400">{stats.count}</span>
              ) : (
                <span className="text-red-400">不可用</span>
              )}{' '}
              条 Runbook
            </span>
            {!stats.available && (
              <span className="flex items-center gap-1 text-amber-400">
                <AlertCircle className="w-3 h-3" />
                搜索服务降级
              </span>
            )}
          </div>
        )}

        <form onSubmit={runSearch} className="mt-4 space-y-3">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="例如：order-service 高 CPU 怎么处理过"
                className="w-full pl-10 pr-4 py-2.5 bg-background border border-border/60 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
              />
            </div>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-5 py-2.5 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              检索
            </button>
          </div>

          <details className="text-sm">
            <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 select-none">
              <Filter className="w-3 h-3" />
              高级筛选
            </summary>
            <div className="grid grid-cols-2 gap-3 mt-3 pl-4">
              <div>
                <label className="text-xs text-muted-foreground">服务名</label>
                <input
                  value={service}
                  onChange={(e) => setService(e.target.value)}
                  placeholder="如 order-service"
                  className="mt-1 w-full px-3 py-1.5 bg-background border border-border/60 rounded text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  最低置信度 (0~1)
                </label>
                <input
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(e.target.value)}
                  placeholder="如 0.7"
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  className="mt-1 w-full px-3 py-1.5 bg-background border border-border/60 rounded text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                />
              </div>
            </div>
          </details>
        </form>
      </div>

      {/* 错误态 */}
      {error && !loading && (
        <div className="glass-card p-5 border-red-400/30 bg-red-400/5">
          <div className="flex items-start gap-2 text-sm">
            <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-red-400 font-medium">检索失败</div>
              <div className="text-muted-foreground mt-1">
                {String((error as { detail?: string; message?: string })?.detail
                  ?? (error as { message?: string })?.message
                  ?? '请稍后重试或检查后端 ChromaDB 状态')}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 结果区 */}
      {loading && (
        <div className="glass-card p-8 flex items-center justify-center gap-3">
          <Loader2 className="w-5 h-5 text-muted-foreground animate-spin" />
          <span className="text-sm text-muted-foreground">正在检索历史 Runbook…</span>
        </div>
      )}

      {!loading && searched && hits.length === 0 && (
        <div className="glass-card p-8 flex flex-col items-center gap-2 text-muted-foreground">
          <AlertCircle className="w-8 h-8 text-muted-foreground/60" />
          <div className="text-sm">没有匹配的 Runbook</div>
          <div className="text-xs">试试更宽泛的关键词，或清空筛选条件</div>
        </div>
      )}

      {!loading && hits.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-muted-foreground px-1">
            <span>共 {hits.length} 条命中（按相关度排序）</span>
          </div>
          {hits.map((hit) => (
            <Link
              key={hit.doc_id}
              to={`/incidents/${hit.incident_id}`}
              className="block glass-card p-4 hover:border-primary/40 transition-colors group"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <BookOpen className="w-4 h-4 text-purple-400 shrink-0" />
                    <h3 className="text-sm font-semibold text-foreground truncate">
                      {highlight(hit.title, query)}
                    </h3>
                    <StatusPill status={hit.status} />
                  </div>

                  <div className="flex items-center flex-wrap gap-2 text-xs text-muted-foreground mb-2">
                    <span>
                      服务:{' '}
                      <span className="text-foreground font-medium">
                        {hit.service || '未知'}
                      </span>
                    </span>
                    <span>·</span>
                    <span>
                      根因:{' '}
                      <span className="text-amber-400 font-medium">
                        {hit.root_cause || 'unknown'}
                      </span>
                    </span>
                    <span>·</span>
                    <span>
                      置信度{' '}
                      <span className="text-foreground">
                        {((hit.confidence ?? 0) * 100).toFixed(1)}%
                      </span>
                    </span>
                    <span>·</span>
                    <span>v{hit.version_number}</span>
                    {hit.published_by && (
                      <>
                        <span>·</span>
                        <span>by {hit.published_by}</span>
                      </>
                    )}
                  </div>

                  {hit.snippet && (
                    <div className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
                      {highlight(hit.snippet, query)}
                    </div>
                  )}

                  {hit.change_note && (
                    <div className="mt-2 text-xs text-muted-foreground">
                      <CheckCircle2 className="w-3 h-3 inline mr-1 text-emerald-400" />
                      {hit.change_note}
                    </div>
                  )}
                </div>

                <div className="flex flex-col items-end gap-2 shrink-0">
                  <ScoreBar score={hit.score} />
                  <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-primary group-hover:translate-x-0.5 transition-all" />
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}

      {/* 空态提示 */}
      {!loading && !searched && (
        <div className="glass-card p-12 flex flex-col items-center gap-3 text-muted-foreground">
          <Sparkles className="w-10 h-10 text-muted-foreground/40" />
          <div className="text-sm">输入关键词检索已发布的 Runbook</div>
          <div className="text-xs text-center max-w-md">
            检索基于 ChromaDB 向量相似度，匹配故障现象、服务名、根因等关键字段。
            <br />
            命中后可直接跳转到对应故障详情页查看完整 Runbook。
          </div>
        </div>
      )}
    </div>
  );
}
