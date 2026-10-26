import { useEffect, useState, type ReactElement } from 'react';
import { BookOpen, FileText, Terminal, AlertCircle, Loader2 } from 'lucide-react';
import { useIncidents } from '@/hooks/useApi';

type RunbookSectionKind =
  | 'overview'
  | 'root_cause'
  | 'impact_chain'
  | 'evidence_summary'
  | 'log_evidence'
  | 'historical_memory'
  | 'actions'
  | 'rollback'
  | 'verification'
  | string;

interface RunbookSectionData {
  heading: string;
  kind: RunbookSectionKind;
  text: string;
  available?: boolean;
  entries_count?: number;
  samples_count?: number;
  actions?: string[];
}

interface RunbookDraft {
  title: string;
  service: string;
  root_cause: string;
  confidence: number;
  generated_at: string;
  sections: RunbookSectionData[];
  markdown: string;
  source?: {
    type: string;
    log_evidence_available?: boolean;
  };
}

/**
 * 简单 Markdown 渲染器（无外部依赖），仅支持本 Runbook 实际产出的语法：
 * - # 一级标题
 * - ## 二级标题
 * - ### 三级标题
 * - 代码块 ```...```
 * - 列表项 "- ..." / "1. ..."
 * - 加粗 **...**
 * - 行内代码 `...`
 * - 段落空行分隔
 */
function renderMarkdown(md: string): ReactElement[] {
  const lines = md.split('\n');
  const out: ReactElement[] = [];
  let codeBuffer: string[] | null = null;
  let key = 0;

  const flushCode = () => {
    if (codeBuffer) {
      out.push(
        <pre
          key={`code-${key++}`}
          className="bg-slate-900/70 border border-slate-700 rounded p-3 overflow-x-auto text-xs font-mono my-2"
        >
          <code>{codeBuffer.join('\n')}</code>
        </pre>
      );
      codeBuffer = null;
    }
  };

  for (const raw of lines) {
    const line = raw;
    if (codeBuffer !== null) {
      if (line.trim().startsWith('```')) {
        flushCode();
      } else {
        codeBuffer.push(line);
      }
      continue;
    }
    if (line.trim().startsWith('```')) {
      flushCode();
      codeBuffer = [];
      continue;
    }
    if (line.startsWith('# ')) {
      out.push(
        <h1 key={key++} className="text-xl font-bold mt-4 mb-2">
          {line.slice(2)}
        </h1>
      );
    } else if (line.startsWith('## ')) {
      out.push(
        <h2 key={key++} className="text-base font-semibold mt-4 mb-2">
          {line.slice(3)}
        </h2>
      );
    } else if (line.startsWith('### ')) {
      out.push(
        <h3 key={key++} className="text-sm font-semibold mt-3 mb-1">
          {line.slice(4)}
        </h3>
      );
    } else if (/^[-*] /.test(line.trim())) {
      out.push(
        <div key={key++} className="text-sm flex gap-2 my-0.5">
          <span className="text-muted-foreground">•</span>
          <span dangerouslySetInnerHTML={{ __html: formatInline(line.trim().slice(2)) }} />
        </div>
      );
    } else if (/^\d+\. /.test(line.trim())) {
      const match = line.trim().match(/^(\d+)\. (.*)/);
      if (match) {
        out.push(
          <div key={key++} className="text-sm flex gap-2 my-0.5">
            <span className="text-muted-foreground">{match[1]}.</span>
            <span dangerouslySetInnerHTML={{ __html: formatInline(match[2]) }} />
          </div>
        );
      }
    } else if (line.trim() === '') {
      out.push(<div key={key++} className="h-2" />);
    } else {
      out.push(
        <p
          key={key++}
          className="text-sm leading-relaxed my-1"
          dangerouslySetInnerHTML={{ __html: formatInline(line) }}
        />
      );
    }
  }
  flushCode();
  return out;
}

function formatInline(text: string): string {
  // 转义 HTML 但保留我们控制的标记
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  return escaped
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-slate-800 text-amber-300 text-xs">$1</code>');
}

interface LogEvidenceBlockProps {
  section: RunbookSectionData;
}

function LogEvidenceBlock({ section }: LogEvidenceBlockProps) {
  if (!section.available) {
    return (
      <div className="flex items-start gap-2 text-sm text-muted-foreground bg-slate-900/40 rounded p-3 border border-slate-800">
        <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
        <div>{section.text}</div>
      </div>
    );
  }
  // 从 markdown 文本里解析 root_cause`xxx` × N 形式
  const causeRegex = /root_cause`([^`]+)`\s*×\s*(\d+)/g;
  const causes: Array<{ cause: string; count: number }> = [];
  let m: RegExpExecArray | null;
  while ((m = causeRegex.exec(section.text)) !== null) {
    causes.push({ cause: m[1], count: parseInt(m[2], 10) });
  }
  const statusLine = section.text.match(/- \*\*状态\*\*: ([^\n]+)/)?.[1] ?? '';
  const corroborates = statusLine.includes('佐证');
  const diverges = statusLine.includes('分歧');

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-slate-900/40 rounded p-2 border border-slate-800">
          <div className="text-xs text-muted-foreground">条目数</div>
          <div className="text-sm font-semibold">{section.entries_count ?? 0}</div>
        </div>
        <div className="bg-slate-900/40 rounded p-2 border border-slate-800">
          <div className="text-xs text-muted-foreground">样本</div>
          <div className="text-sm font-semibold">{section.samples_count ?? 0}</div>
        </div>
        <div className={`rounded p-2 border ${
          corroborates
            ? 'bg-emerald-900/20 border-emerald-700/40'
            : diverges
              ? 'bg-amber-900/20 border-amber-700/40'
              : 'bg-slate-900/40 border-slate-800'
        }`}>
          <div className="text-xs text-muted-foreground">一致性</div>
          <div className={`text-sm font-semibold ${
            corroborates ? 'text-emerald-400' : diverges ? 'text-amber-400' : ''
          }`}>
            {corroborates ? '✅ 佐证' : diverges ? '⚠️ 分歧' : 'ℹ️ 仅记录'}
          </div>
        </div>
      </div>

      {causes.length > 0 && (
        <div>
          <div className="text-xs text-muted-foreground mb-1">根因签名投票</div>
          <div className="flex flex-wrap gap-1.5">
            {causes.map((c) => (
              <span
                key={c.cause}
                className="px-2 py-0.5 bg-amber-400/10 text-amber-400 rounded text-xs"
              >
                {c.cause} × {c.count}
              </span>
            ))}
          </div>
        </div>
      )}

      <details className="bg-slate-900/40 rounded border border-slate-800">
        <summary className="cursor-pointer text-xs text-muted-foreground px-3 py-2 hover:text-foreground">
          展开 Markdown 源文
        </summary>
        <div className="px-3 pb-3">{renderMarkdown(section.text)}</div>
      </details>
    </div>
  );
}

interface RunbookSectionProps {
  incidentId: string;
}

export function RunbookSection({ incidentId }: RunbookSectionProps) {
  const { getIncidentRunbook } = useIncidents();
  const [runbook, setRunbook] = useState<RunbookDraft | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetch = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = (await getIncidentRunbook(incidentId)) as
          | { runbook?: RunbookDraft }
          | null;
        if (cancelled) return;
        if (result?.runbook) {
          setRunbook(result.runbook);
        } else {
          setError('Runbook 尚未生成（RCA 未完成或失败）');
        }
      } catch (e) {
        if (cancelled) return;
        const msg = (e as { detail?: string; message?: string })?.detail
          || (e as { message?: string })?.message
          || '获取 Runbook 失败';
        setError(msg);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetch();
    return () => {
      cancelled = true;
    };
  }, [incidentId, getIncidentRunbook]);

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-purple-400" />
          <h3 className="text-base font-semibold text-foreground">Runbook 草案</h3>
          {runbook && (
            <span className="text-xs text-muted-foreground">
              置信度 {(runbook.confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
        {runbook?.source?.log_evidence_available && (
          <span className="text-xs px-2 py-0.5 bg-emerald-400/10 text-emerald-400 rounded">
            含 Loki 日志证据
          </span>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
          <Loader2 className="w-4 h-4 animate-spin" />
          正在加载 Runbook…
        </div>
      )}

      {error && !loading && (
        <div className="flex items-start gap-2 text-sm text-muted-foreground bg-slate-900/40 rounded p-3 border border-slate-800">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <div>{error}</div>
        </div>
      )}

      {runbook && !loading && !error && (
        <div className="space-y-5">
          <div className="flex items-center justify-between text-xs text-muted-foreground border-b border-slate-800 pb-2">
            <span>
              <FileText className="w-3 h-3 inline mr-1" />
              {runbook.title}
            </span>
            <span>生成于 {new Date(runbook.generated_at).toLocaleString('zh-CN')}</span>
          </div>

          {runbook.sections.map((sec, idx) => (
            <div key={`${sec.kind}-${idx}`} className="space-y-2">
              <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
                {sec.kind === 'rollback' && <Terminal className="w-3.5 h-3.5 text-orange-400" />}
                {sec.heading}
              </h4>
              {sec.kind === 'log_evidence' ? (
                <LogEvidenceBlock section={sec} />
              ) : sec.kind === 'rollback' || sec.kind === 'verification' ? (
                <div className="space-y-1">{renderMarkdown(sec.text)}</div>
              ) : (
                <div className="space-y-1 text-sm">{renderMarkdown(sec.text)}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
