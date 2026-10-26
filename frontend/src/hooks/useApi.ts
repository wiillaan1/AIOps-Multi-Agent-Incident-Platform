import { useState, useCallback, useRef, useMemo } from 'react';

const API_BASE = '/api/v1';

interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

interface UseApiReturn<T> extends ApiState<T> {
  get: (url: string) => Promise<T | null>;
  post: (url: string, body: Record<string, unknown>) => Promise<T | null>;
}

/**
 * Generic API hook — returns stable get/post functions and reactive data/loading/error state.
 *
 * Key design: get/post are stabilized via useRef so they never cause downstream
 * useCallback/useEffect churn. Only data/loading/error trigger re-renders.
 */
export function useApi<T = unknown>(): UseApiReturn<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Keep request ID to avoid stale responses
  const reqIdRef = useRef(0);

  // Stable fetch core — never changes, so downstream deps are safe
  const fetchRef = useRef(async (url: string, options: RequestInit = {}): Promise<T | null> => {
    const reqId = ++reqIdRef.current;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}${url}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error ${response.status}: ${errorText}`);
      }

      const result = (await response.json()) as T;

      // Only update state if this is still the latest request
      if (reqId === reqIdRef.current) {
        setData(result);
      }
      return result;
    } catch (err) {
      const errorObj = err instanceof Error ? err : new Error('Unknown error');
      if (reqId === reqIdRef.current) {
        setError(errorObj);
      }
      console.warn('[API]', errorObj.message);
      return null;
    } finally {
      if (reqId === reqIdRef.current) {
        setLoading(false);
      }
    }
  });

  // Stable function references — memoized once, never change
  const get = useCallback(
    (url: string) => fetchRef.current(url, { method: 'GET' }),
    [] // ✅ empty deps = stable forever
  );

  const post = useCallback(
    (url: string, body: Record<string, unknown>) =>
      fetchRef.current(url, { method: 'POST', body: JSON.stringify(body) }),
    [] // ✅ empty deps = stable forever
  );

  // Return stable get/post + reactive state
  return useMemo(() => ({ data, loading, error, get, post }), [data, loading, error, get, post]);
}

// ============================================================
// Convenience hooks — stable callback references
// ============================================================

export function useIncidents() {
  const api = useApi<{
    items: Array<Record<string, unknown>>;
    total: number;
  }>();

  // Extract stable get/post from api
  const { get, post, data, loading, error } = api;

  const listIncidents = useCallback(
    (params?: { severity?: string; status?: string; page?: number; limit?: number }) => {
      const queryParams = new URLSearchParams();
      if (params?.severity) queryParams.append('severity', params.severity);
      if (params?.status) queryParams.append('status', params.status);
      if (params?.page) queryParams.append('page', String(params.page));
      if (params?.limit) queryParams.append('limit', String(params.limit));
      return get(`/incidents?${queryParams.toString()}`);
    },
    [get] // ✅ get is stable
  );

  const getIncident = useCallback(
    (id: string) => get(`/incidents/${id}`),
    [get]
  );

  const getIncidentRunbook = useCallback(
    (id: string) => get(`/incidents/${id}/runbook`),
    [get]
  );

  const triggerIncident = useCallback(
    (body: { service: string; metric: string; severity: string; value?: number; threshold?: number }) =>
      post('/incidents/trigger', {
        ...body,
        value: body.value ?? 95,
        threshold: body.threshold ?? 80,
        operator: '>',
        source: 'frontend',
        labels: { tier: 'critical' },
        annotations: {},
      }),
    [post] // ✅ post is stable
  );

  // ----- W9 调查验证闭环 -----
  const getInvestigation = useCallback(
    (id: string) => get(`/incidents/${id}/investigation`),
    [get]
  );

  const getInvestigationEvidence = useCallback(
    (id: string, params?: { assertion_id?: string }) => {
      const queryParams = new URLSearchParams();
      if (params?.assertion_id) queryParams.append('assertion_id', params.assertion_id);
      const qs = queryParams.toString();
      return get(`/incidents/${id}/investigation/evidence${qs ? `?${qs}` : ''}`);
    },
    [get]
  );

  const submitExecutionReceipt = useCallback(
    (
      id: string,
      payload: {
        idempotency_key: string;
        action_id: string;
        playbook_id: string;
        target_resource: string;
        status: 'succeeded' | 'failed' | 'rejected';
        receipt_id?: string;
        completed_at?: string | null;
        summary?: Record<string, unknown>;
      }
    ) => post(`/incidents/${id}/execution-receipts`, { ...payload, incident_id: id }),
    [post]
  );

  const requestRecoveryVerification = useCallback(
    (id: string) => post(`/incidents/${id}/recovery-verification`, {}),
    [post]
  );

  const listRootCauseProfiles = useCallback(
    () => get('/root-cause-profiles'),
    [get]
  );

  return {
    data,
    loading,
    error,
    listIncidents,
    getIncident,
    getIncidentRunbook,
    triggerIncident,
    getInvestigation,
    getInvestigationEvidence,
    submitExecutionReceipt,
    requestRecoveryVerification,
    listRootCauseProfiles,
  };
}

export function useAgents() {
  const { get, data, loading, error } = useApi<{
    items: Array<Record<string, unknown>>;
    total: number;
  }>();

  const listAgents = useCallback(() => get('/agents'), [get]);
  const getAgentStatus = useCallback((id: string) => get(`/agents/${id}/status`), [get]);

  return { data, loading, error, listAgents, getAgentStatus };
}

export function useEvaluations() {
  const { get, post, data, loading, error } = useApi<{
    items: Array<Record<string, unknown>>;
    total: number;
  }>();

  const listEvaluations = useCallback(() => get('/evaluations'), [get]);
  const getEvaluation = useCallback((id: string) => get(`/evaluations/${id}`), [get]);
  const runEvaluation = useCallback(
    (evalType: string) => post('/evaluations/run', { eval_type: evalType }),
    [post]
  );

  return { data, loading, error, listEvaluations, getEvaluation, runEvaluation };
}

export function useTopology() {
  const { get, data, loading, error } = useApi<{
    nodes: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
  }>();

  const getTopology = useCallback(() => get('/topology'), [get]);

  return { data, loading, error, getTopology };
}

export function useMemory() {
  const { get, post, data, loading, error } = useApi<{
    results: Array<Record<string, unknown>>;
    total: number;
  }>();

  const searchMemory = useCallback(
    (query: string) => get(`/memory/search?query=${encodeURIComponent(query)}`),
    [get]
  );

  const storeMemory = useCallback(
    (content: string, memoryType: string = 'observation', tags: string[] = []) =>
      post('/memory/store', { content, memory_type: memoryType, tags }),
    [post]
  );

  return { data, loading, error, searchMemory, storeMemory };
}

export function useRunbookSearch() {
  const api = useApi<{
    query: string;
    top_k: number;
    filters: Record<string, unknown>;
    count: number;
    hits: Array<Record<string, unknown>>;
  }>();
  const { get, data, loading, error } = api;

  const searchRunbooks = useCallback(
    (params: { q: string; top_k?: number; service?: string; min_confidence?: number }) => {
      const queryParams = new URLSearchParams();
      queryParams.append('q', params.q);
      if (params.top_k) queryParams.append('top_k', String(params.top_k));
      if (params.service) queryParams.append('service', params.service);
      if (typeof params.min_confidence === 'number') {
        queryParams.append('min_confidence', String(params.min_confidence));
      }
      return get(`/runbooks/search?${queryParams.toString()}`);
    },
    [get]
  );

  const getRunbookStats = useCallback(() => get('/runbooks/stats'), [get]);

  return { data, loading, error, searchRunbooks, getRunbookStats };
}
