import { z } from "zod";

export const severityEnum = z.enum(["info", "warning", "critical"]);
export const verdictEnum = z.enum(["READY", "NEEDS ATTENTION", "NOT READY"]);
export const toolEnum = z.enum([
  "Leakage Detector",
  "Train-Test Contamination Detector",
  "Bias Detector",
  "Spurious Correlation Detector",
  "Feature Utility Detector",
  "Planner",
  "Critic",
  "Report Generator"
]);

export const findingSchema = z.object({
  id: z.string(),
  severity: severityEnum,
  tool: z.string(),
  message: z.string(),
  evidence: z.any().optional(), // JSON evidence
});

export const recommendationSchema = z.object({
  id: z.string(),
  priority: z.boolean(),
  text: z.string(),
});

export const criticAssessmentSchema = z.object({
  confidence: z.number().min(0).max(1),
  needsRecheck: z.boolean(),
  message: z.string(),
});

export const pipelineStageSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: z.enum(["pending", "running", "completed", "skipped", "error"]),
  skipReason: z.string().optional(),
  runtimeMs: z.number().optional(),
  criticAssessment: criticAssessmentSchema.optional(),
  liveMessages: z.array(z.string()).optional(),
});

export const executionTimelineItemSchema = z.object({
  tool: z.string(),
  startTime: z.string(),
  endTime: z.string(),
  durationMs: z.number(),
  status: z.string(),
});

export const auditReportSchema = z.object({
  id: z.string(),
  datasetShape: z.object({ rows: z.number(), columns: z.number() }),
  targetColumn: z.string(),
  verdict: verdictEnum,
  readinessScore: z.number().min(0).max(100),
  metrics: z.object({
    totalFindings: z.number(),
    critical: z.number(),
    warning: z.number(),
    confidence: z.number(),
  }),
  findings: z.array(findingSchema),
  recommendations: z.array(recommendationSchema),
  timeline: z.array(executionTimelineItemSchema),
  rawData: z.any(),
});

export const auditStartRequestSchema = z.object({
  filename: z.string(),
  targetColumn: z.string(),
  rows: z.number(),
  columns: z.number(),
});

export const auditStartResponseSchema = z.object({
  id: z.string(),
  message: z.string(),
});

export const auditStatusSchema = z.object({
  id: z.string(),
  isComplete: z.boolean(),
  progressPercentage: z.number().min(0).max(100),
  currentStage: z.string().optional(),
  stages: z.array(pipelineStageSchema),
  optimizationNotice: z.string().optional(),
});

// Extracted Types
export type Finding = z.infer<typeof findingSchema>;
export type Recommendation = z.infer<typeof recommendationSchema>;
export type CriticAssessment = z.infer<typeof criticAssessmentSchema>;
export type PipelineStage = z.infer<typeof pipelineStageSchema>;
export type ExecutionTimelineItem = z.infer<typeof executionTimelineItemSchema>;
export type AuditReport = z.infer<typeof auditReportSchema>;
export type AuditStatus = z.infer<typeof auditStatusSchema>;
export type AuditStartRequest = z.infer<typeof auditStartRequestSchema>;
