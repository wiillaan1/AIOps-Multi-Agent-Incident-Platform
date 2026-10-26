export type Severity = 'critical' | 'high' | 'medium' | 'low';
export type IncidentStatus =
  | 'pending' | 'new' | 'acknowledged'
  | 'detecting' | 'analyzing'
  | 'rca_in_progress' | 'rca_completed'
  | 'healing' | 'awaiting_approval'
  | 'resolved' | 'failed' | 'escalated' | 'closed' | 'cancelled';
export type AgentStatus = 'running' | 'stopped' | 'error';

export interface Alert {
  is_anomaly: boolean;
  confidence: number;
  algorithms_voted: string[];
}

export interface RCA {
  root_cause: string;
  confidence: number;
  impact_chain: string[];
  suggested_actions: string[];
}

export interface Heal {
  action: string;
  level: 'L0' | 'L1' | 'L2';
  dry_run_result: string;
  blast_radius: number;
}

export interface Change {
  approval_status: string;
  risk_score: number;
  approver?: string;
}

export interface Incident {
  id: string;
  service: string;
  metric: string;
  severity: Severity;
  status: IncidentStatus;
  alert?: Alert;
  rca?: RCA;
  heal?: Heal;
  change?: Change;
  created_at: string;
  updated_at: string;
}

export interface Agent {
  id: string;
  name: string;
  description: string;
  status: AgentStatus;
  last_activity: string;
  type?: string;
}

export interface EvalMetric {
  name: string;
  score: number;
  weight: number;
}

export interface EvaluationResult {
  id: string;
  dimension: string;
  overall_score: number;
  metrics: EvalMetric[];
  timestamp: string;
}

export interface MemoryEntry {
  id: string;
  key: string;
  value: string;
  type: 'incident' | 'playbook' | 'config' | 'knowledge';
  created_at: string;
  importance: number;
  tags?: string[];
}

export interface TopologyNode {
  id: string;
  name: string;
  service: string;
  status: 'healthy' | 'warning' | 'critical' | 'unknown';
  dependencies: string[];
  metrics?: {
    cpu: number;
    memory: number;
    latency: number;
    error_rate: number;
  };
}

export interface TopologyEdge {
  source: string;
  target: string;
  type: 'http' | 'rpc' | 'db' | 'mq';
  latency?: number;
}

export interface TopologyData {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
}

export interface WebSocketMessage {
  type: 'incident_update' | 'agent_status' | 'evaluation_result' | 'heartbeat';
  payload: Record<string, unknown>;
  timestamp: string;
}

export interface IncidentStats {
  total: number;
  resolved: number;
  processing: number;
  failed: number;
  avg_mttr: number;
}

export interface AgentStats {
  total: number;
  running: number;
  stopped: number;
  error: number;
}

export interface TrendPoint {
  timestamp: string;
  score: number;
  dimension: string;
}

// ============================================================
// W9 调查验证闭环（InvestigationLoopEngine）类型
// 与 backend app/models/investigation.py 的 JSON 序列化对齐
// ============================================================

export type InvestigationState =
  | 'pending'
  | 'investigating'
  | 'replanning'
  | 'awaiting_execution'
  | 'recovery_observing'
  | 'recovered'
  | 'heal_failed'
  | 'manual_escalation';

export type VerificationStatus =
  | 'confirmed'
  | 'contradicted'
  | 'inconclusive'
  | 'manual_only';

export type RecoveryOutcome = 'recovered' | 'degraded' | 'failed' | 'inconclusive';

export type AssertionOutcome = 'pass' | 'fail' | 'counter' | 'unavailable';

export type EvidenceSource = 'prometheus' | 'loki' | 'health' | 'guardrail';

export interface RootCauseHypothesis {
  hypothesis_id: string;
  root_cause: string;
  confidence: number;
  rca_round: number;
  profile_version: string | null;
  evidence_refs: string[];
  summary: Record<string, unknown>;
  created_at: string;
}

export interface EvidenceRecord {
  evidence_id: string;
  incident_id: string;
  assertion_id: string;
  source: EvidenceSource;
  outcome: AssertionOutcome;
  observed_at: string;
  query_summary: Record<string, unknown>;
  summary: Record<string, unknown>;
  score: number | null;
}

export interface HealExecutionReceipt {
  receipt_id: string;
  incident_id: string;
  action_id: string;
  idempotency_key: string;
  playbook_id: string;
  target_resource: string;
  status: 'succeeded' | 'failed' | 'rejected';
  received_at: string;
  completed_at: string | null;
  summary: Record<string, unknown>;
}

export interface RecoveryVerification {
  verification_id: string;
  incident_id: string;
  receipt_id: string;
  receipt_status: 'succeeded';
  outcome: RecoveryOutcome;
  verified_at: string;
  signals: Array<Record<string, unknown>>;
  unavailable_sources: EvidenceSource[];
  summary: Record<string, unknown>;
}

/** Backend snapshot for /incidents/{id}/investigation (run + evidence merged). */
export interface InvestigationSnapshot {
  incident_id: string;
  run_id: string;
  state: InvestigationState;
  verification_status: VerificationStatus;
  rca_round: number;
  heal_attempt: number;
  hypotheses: RootCauseHypothesis[];
  evidence_refs: string[];
  receipts: HealExecutionReceipt[];
  recovery_verifications: RecoveryVerification[];
  audit_summary: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  evidence: EvidenceRecord[];
}

export interface InvestigationEvidenceResponse {
  incident_id: string;
  evidence: EvidenceRecord[];
  count: number;
}

export interface RootCauseProfileSummary {
  root_cause: string;
  version: string;
  manual_only: boolean;
}

export interface RootCauseProfileListResponse {
  profiles: RootCauseProfileSummary[];
  count: number;
}

/** Expanded WebSocket event union for investigation state changes. */
export type InvestigationEventType =
  | 'investigation_state_changed'
  | 'investigation_evidence_recorded'
  | 'execution_receipt_received'
  | 'recovery_verification_updated'
  | 'manual_escalation_required';
