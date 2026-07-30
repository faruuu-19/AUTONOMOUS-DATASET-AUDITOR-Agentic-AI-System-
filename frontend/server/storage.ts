import { 
  AuditReport, 
  AuditStatus, 
  AuditStartRequest,
  PipelineStage
} from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  startAudit(request: AuditStartRequest): Promise<string>;
  getAuditStatus(id: string): Promise<AuditStatus | undefined>;
  getAuditReport(id: string): Promise<AuditReport | undefined>;
}

export class MemStorage implements IStorage {
  private audits: Map<string, {
    request: AuditStartRequest;
    startTime: number;
  }> = new Map();

  async startAudit(request: AuditStartRequest): Promise<string> {
    const id = randomUUID();
    this.audits.set(id, { request, startTime: Date.now() });
    return id;
  }

  async getAuditStatus(id: string): Promise<AuditStatus | undefined> {
    const audit = this.audits.get(id);
    if (!audit) return undefined;

    const elapsed = Date.now() - audit.startTime;
    // Simulate a 15-second audit process
    const totalDuration = 15000;
    let progress = Math.min(100, Math.floor((elapsed / totalDuration) * 100));
    const isComplete = progress >= 100;

    const stages: PipelineStage[] = [
      {
        id: "s1",
        name: "Planner",
        status: elapsed > 1000 ? "completed" : "running",
        runtimeMs: 1200,
        liveMessages: ["Analyzing dataset shape...", "Selecting optimal tools..."]
      },
      {
        id: "s2",
        name: "Leakage Detector",
        status: elapsed > 3000 ? "completed" : (elapsed > 1000 ? "running" : "pending"),
        runtimeMs: 2500,
        liveMessages: elapsed > 1000 ? ["Scanning for target correlation...", "Checking timestamp overlaps..."] : []
      },
      {
        id: "s3",
        name: "Bias Detector",
        status: elapsed > 7000 ? "completed" : (elapsed > 3000 ? "running" : "pending"),
        runtimeMs: 4000,
        liveMessages: elapsed > 3000 ? ["Analyzing categorical distribution...", "Calculating disparate impact..."] : [],
        criticAssessment: elapsed > 7000 ? { confidence: 0.4, needsRecheck: true, message: "Sample size too small for robust bias metric." } : undefined
      },
      {
        id: "s4",
        name: "Adaptive Recheck",
        status: elapsed > 11000 ? "completed" : (elapsed > 7000 ? "running" : "pending"),
        runtimeMs: 3500,
        liveMessages: elapsed > 7000 ? ["Bootstrapping additional samples...", "Re-evaluating bias metrics..."] : [],
        skipReason: elapsed < 7000 ? "Waiting for initial detectors" : undefined
      },
      {
        id: "s5",
        name: "Report Generation",
        status: isComplete ? "completed" : (elapsed > 11000 ? "running" : "pending"),
        runtimeMs: 1000,
        liveMessages: elapsed > 11000 ? ["Aggregating findings...", "Compiling JSON report..."] : []
      }
    ];

    let currentStage = stages.find(s => s.status === "running")?.name;
    if (isComplete) currentStage = "Complete";

    return {
      id,
      isComplete,
      progressPercentage: progress,
      currentStage,
      stages,
      optimizationNotice: audit.request.rows > 100000 ? "Large dataset detected. Using adaptive 10% sampling." : undefined
    };
  }

  async getAuditReport(id: string): Promise<AuditReport | undefined> {
    const audit = this.audits.get(id);
    if (!audit) return undefined;

    const status = await this.getAuditStatus(id);
    if (!status || !status.isComplete) return undefined;

    // Return realistic seeded data
    return {
      id,
      datasetShape: { rows: audit.request.rows, columns: audit.request.columns },
      targetColumn: audit.request.targetColumn,
      verdict: "NEEDS ATTENTION",
      readinessScore: 68,
      metrics: {
        totalFindings: 12,
        critical: 2,
        warning: 4,
        confidence: 0.85
      },
      findings: [
        {
          id: "f1",
          severity: "critical",
          tool: "Leakage Detector",
          message: "Feature 'transaction_date' is highly correlated with target and occurs after the target event.",
          evidence: { correlation: 0.98, time_overlap: true }
        },
        {
          id: "f2",
          severity: "critical",
          tool: "Train-Test Contamination Detector",
          message: "Potential identifier 'customer_id' has overlap between historical periods, risking data bleed.",
          evidence: { overlap_ratio: 0.15 }
        },
        {
          id: "f3",
          severity: "warning",
          tool: "Bias Detector",
          message: "Minority category in 'region' exhibits 25% lower prediction confidence.",
          evidence: { disparate_impact: 0.75 }
        },
        {
          id: "f4",
          severity: "info",
          tool: "Feature Utility Detector",
          message: "5 features have zero variance and can be safely removed.",
          evidence: { features: ["col_A", "col_B", "col_C", "col_D", "col_E"] }
        }
      ],
      recommendations: [
        { id: "r1", priority: true, text: "Drop 'transaction_date' before training." },
        { id: "r2", priority: true, text: "Implement group-based splitting on 'customer_id' to prevent train/test bleed." },
        { id: "r3", priority: false, text: "Consider upsampling techniques for underrepresented 'region' categories." },
        { id: "r4", priority: false, text: "Drop zero-variance features." }
      ],
      timeline: [
        { tool: "Planner", startTime: "00:00.00", endTime: "00:01.20", durationMs: 1200, status: "success" },
        { tool: "Leakage Detector", startTime: "00:01.20", endTime: "00:03.70", durationMs: 2500, status: "success" },
        { tool: "Bias Detector", startTime: "00:03.70", endTime: "00:07.70", durationMs: 4000, status: "warning" },
        { tool: "Adaptive Recheck", startTime: "00:07.70", endTime: "00:11.20", durationMs: 3500, status: "success" },
        { tool: "Report Generator", startTime: "00:11.20", endTime: "00:12.20", durationMs: 1000, status: "success" },
      ],
      rawData: {
        raw_audit_log: "...",
        meta_parameters: { threshold: 0.05, sampling_rate: 0.1 }
      }
    };
  }
}

export const storage = new MemStorage();
