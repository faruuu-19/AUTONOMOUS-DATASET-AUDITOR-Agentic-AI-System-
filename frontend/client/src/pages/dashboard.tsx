import { useParams } from "wouter";
import { Layout } from "../components/layout";
import { useAuditReport } from "../hooks/use-audit";
import { ScoreDial } from "../components/score-dial";
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, Cell
} from "recharts";
import { 
  Download, FileJson, AlertOctagon, CheckCircle2, ShieldAlert, Zap, Database
} from "lucide-react";
import { cn } from "../lib/utils";
import { useEffect, useState } from "react";
import { useToast } from "../hooks/use-toast";

export function Dashboard() {
  const params = useParams();
  const { data: report, isLoading, isError } = useAuditReport(params.id || "");
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState<"findings" | "recommendations" | "timeline">("findings");
  const [expandedEvidence, setExpandedEvidence] = useState<Record<string, boolean>>({});
  const [isExportingJson, setIsExportingJson] = useState(false);
  const [isExportingCsv, setIsExportingCsv] = useState(false);

  useEffect(() => {
    setExpandedEvidence({});
  }, [report?.id]);

  if (isLoading) {
    return (
      <Layout>
        <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin glow-cyan" />
          <div className="text-xl font-display font-semibold text-primary glow-text">Compiling Final Report...</div>
        </div>
      </Layout>
    );
  }

  if (isError || !report) {
    return (
      <Layout>
        <div className="text-center text-destructive mt-20 font-mono">Error 404: Report Data Not Found.</div>
      </Layout>
    );
  }

  const isReady = report.verdict === "READY";
  const isNeedsAttention = report.verdict === "NEEDS ATTENTION";
  
  const verdictColor = isReady ? "text-success" : isNeedsAttention ? "text-warning" : "text-destructive";
  const verdictBg = isReady ? "bg-success/10 border-success/30 glow-teal" : isNeedsAttention ? "bg-warning/10 border-warning/30 glow-warning" : "bg-destructive/10 border-destructive/30 glow-destructive";

  const timelineData = report.timeline.map(t => ({
    name: t.tool.replace(' Detector', ''),
    duration: t.durationMs,
    status: t.status
  }));
  const evidenceFindingIds = report.findings.filter((f) => !!f.evidence).map((f) => f.id);
  const allEvidenceExpanded =
    evidenceFindingIds.length > 0 && evidenceFindingIds.every((id) => !!expandedEvidence[id]);

  const toggleEvidence = (findingId: string) => {
    setExpandedEvidence((prev) => ({
      ...prev,
      [findingId]: !prev[findingId],
    }));
  };

  const toggleAllEvidence = () => {
    if (allEvidenceExpanded) {
      setExpandedEvidence({});
      return;
    }

    const nextState: Record<string, boolean> = {};
    evidenceFindingIds.forEach((id) => {
      nextState[id] = true;
    });
    setExpandedEvidence(nextState);
  };

  const getDownloadFilename = (header: string | null, fallback: string): string => {
    if (!header) return fallback;

    const utf8Match = header.match(/filename\*=UTF-8''([^;]+)/i);
    if (utf8Match && utf8Match[1]) {
      return decodeURIComponent(utf8Match[1]);
    }

    const regularMatch = header.match(/filename="?([^"]+)"?/i);
    if (regularMatch && regularMatch[1]) {
      return regularMatch[1];
    }

    return fallback;
  };

  const downloadFile = async (endpoint: string, fallbackName: string) => {
    const response = await fetch(endpoint, { credentials: "include" });
    if (!response.ok) {
      throw new Error("Download failed");
    }

    const blob = await response.blob();
    const fileUrl = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = fileUrl;
    anchor.download = getDownloadFilename(response.headers.get("content-disposition"), fallbackName);
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.URL.revokeObjectURL(fileUrl);
  };

  const reportId = params.id || report.id;

  const handleExportJson = async () => {
    if (!reportId) return;

    setIsExportingJson(true);
    try {
      await downloadFile(`/api/audit/${reportId}/export/json`, `audit_report_${reportId}.json`);
    } catch (_error) {
      toast({
        title: "Export failed",
        description: "Could not download JSON report.",
        variant: "destructive",
      });
    } finally {
      setIsExportingJson(false);
    }
  };

  const handleExportCleanedCsv = async () => {
    if (!reportId) return;

    setIsExportingCsv(true);
    try {
      await downloadFile(`/api/audit/${reportId}/export/cleaned-csv`, `cleaned_dataset_${reportId}.csv`);
    } catch (_error) {
      toast({
        title: "Export failed",
        description: "Could not download cleaned CSV.",
        variant: "destructive",
      });
    } finally {
      setIsExportingCsv(false);
    }
  };

  return (
    <Layout>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-display font-bold mb-2">Audit Results</h1>
          <p className="text-muted-foreground flex items-center gap-3">
            <span className="px-2 py-1 rounded bg-black/50 border border-white/10 font-mono text-xs">ID: {report.id}</span>
            <span>Target: <strong className="text-foreground">{report.targetColumn}</strong></span>
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleExportJson}
            disabled={isExportingJson || isExportingCsv}
            className="px-4 py-2 rounded-lg bg-black/50 border border-white/10 hover:border-primary/50 hover:text-primary transition-all flex items-center gap-2 text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <FileJson className="w-4 h-4" />
            {isExportingJson ? "Exporting..." : "Export JSON"}
          </button>
          <button
            onClick={handleExportCleanedCsv}
            disabled={isExportingJson || isExportingCsv}
            className="px-4 py-2 rounded-lg bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 hover:glow-cyan transition-all flex items-center gap-2 text-sm font-bold disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            {isExportingCsv ? "Preparing..." : "Cleaned CSV"}
          </button>
        </div>
      </div>

      {/* Top Grid: Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Score & Verdict Card */}
        <div className="glass-panel p-6 rounded-2xl flex flex-col items-center justify-center md:col-span-1">
          <ScoreDial score={report.readinessScore} verdict={report.verdict} />
          <div className={cn("mt-6 px-6 py-2 rounded-full border text-sm font-bold tracking-widest uppercase", verdictBg, verdictColor)}>
            {report.verdict}
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="md:col-span-2 grid grid-cols-2 gap-4">
          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
            <div className="text-muted-foreground text-sm font-semibold flex items-center gap-2 mb-4">
              <ShieldAlert className="w-4 h-4 text-critical" /> Critical Findings
            </div>
            <div className="text-5xl font-display font-bold text-destructive glow-text">
              {report.metrics.critical}
            </div>
          </div>
          
          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
            <div className="text-muted-foreground text-sm font-semibold flex items-center gap-2 mb-4">
              <AlertOctagon className="w-4 h-4 text-warning" /> Warnings
            </div>
            <div className="text-5xl font-display font-bold text-warning glow-text">
              {report.metrics.warning}
            </div>
          </div>
          
          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
            <div className="text-muted-foreground text-sm font-semibold flex items-center gap-2 mb-4">
              <CheckCircle2 className="w-4 h-4 text-success" /> Confidence
            </div>
            <div className="text-5xl font-display font-bold text-success glow-text">
              {(report.metrics.confidence * 100).toFixed(0)}%
            </div>
          </div>
          
          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between bg-gradient-to-br from-card/80 to-primary/5">
            <div className="text-muted-foreground text-sm font-semibold flex items-center gap-2 mb-4">
              <Database className="w-4 h-4 text-primary" /> Dataset Shape
            </div>
            <div className="text-2xl font-mono text-primary">
              {report.datasetShape.rows.toLocaleString()} <span className="text-sm text-muted-foreground">rows</span>
              <br/>
              {report.datasetShape.columns.toLocaleString()} <span className="text-sm text-muted-foreground">cols</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs Section */}
      <div className="glass-panel rounded-2xl overflow-hidden min-h-[500px]">
        <div className="flex border-b border-white/10 bg-black/20">
          <button 
            className={cn("px-8 py-4 font-semibold text-sm transition-colors border-b-2", activeTab === "findings" ? "border-primary text-primary bg-primary/5" : "border-transparent text-muted-foreground hover:text-foreground")}
            onClick={() => setActiveTab("findings")}
          >
            Detailed Findings
          </button>
          <button 
            className={cn("px-8 py-4 font-semibold text-sm transition-colors border-b-2", activeTab === "recommendations" ? "border-accent text-accent bg-accent/5" : "border-transparent text-muted-foreground hover:text-foreground")}
            onClick={() => setActiveTab("recommendations")}
          >
            Recommendations
          </button>
          <button 
            className={cn("px-8 py-4 font-semibold text-sm transition-colors border-b-2", activeTab === "timeline" ? "border-secondary text-secondary bg-secondary/5" : "border-transparent text-muted-foreground hover:text-foreground")}
            onClick={() => setActiveTab("timeline")}
          >
            Execution Timeline
          </button>
        </div>

        <div className="p-6">
          {activeTab === "findings" && (
            <div className="space-y-4">
              {evidenceFindingIds.length > 0 && (
                <div className="flex justify-end">
                  <button
                    onClick={toggleAllEvidence}
                    className="px-3 py-1.5 rounded-md border border-white/15 text-xs font-semibold text-muted-foreground hover:text-foreground hover:border-primary/40 transition-colors"
                  >
                    {allEvidenceExpanded ? "Collapse All Details" : "Expand All Details"}
                  </button>
                </div>
              )}
              {report.findings.length === 0 ? (
                <div className="text-center text-muted-foreground py-12">No findings detected. Your dataset is exceptionally clean.</div>
              ) : (
                report.findings.map(finding => (
                  <div key={finding.id} className={cn(
                    "p-5 rounded-xl border flex flex-col md:flex-row gap-4",
                    finding.severity === "critical" ? "bg-destructive/5 border-destructive/30" : 
                    finding.severity === "warning" ? "bg-warning/5 border-warning/30" : 
                    "bg-secondary/5 border-secondary/30"
                  )}>
                    <div className="md:w-48 flex-shrink-0">
                      <div className={cn(
                        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider mb-2",
                        finding.severity === "critical" ? "bg-destructive/20 text-destructive" : 
                        finding.severity === "warning" ? "bg-warning/20 text-warning" : 
                        "bg-secondary/20 text-secondary"
                      )}>
                        {finding.severity === "critical" && <AlertOctagon className="w-3 h-3" />}
                        {finding.severity === "warning" && <ShieldAlert className="w-3 h-3" />}
                        {finding.severity === "info" && <CheckCircle2 className="w-3 h-3" />}
                        {finding.severity}
                      </div>
                      <div className="text-sm text-muted-foreground font-mono">{finding.tool}</div>
                    </div>
                    <div className="flex-1">
                      <p className="text-foreground text-sm leading-relaxed">{finding.message}</p>
                      {finding.evidence && (
                        <div className="mt-4">
                          <button
                            onClick={() => toggleEvidence(finding.id)}
                            className="mb-3 px-3 py-1.5 rounded-md border border-white/15 text-xs font-semibold text-muted-foreground hover:text-foreground hover:border-primary/40 transition-colors"
                          >
                            {expandedEvidence[finding.id] ? "Hide Details" : "Show Details"}
                          </button>
                          {expandedEvidence[finding.id] && (
                            <div className="p-3 bg-black/60 rounded-md border border-white/5 font-mono text-xs text-muted-foreground overflow-x-auto">
                              <pre>{JSON.stringify(finding.evidence, null, 2)}</pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {activeTab === "recommendations" && (
            <div className="space-y-4">
              {report.recommendations.map(rec => (
                <div key={rec.id} className={cn(
                  "p-5 rounded-xl border flex items-start gap-4 transition-all hover:bg-white/5",
                  rec.priority ? "border-accent/40 bg-accent/5 glow-teal" : "border-white/10"
                )}>
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5",
                    rec.priority ? "bg-accent/20 text-accent" : "bg-white/10 text-muted-foreground"
                  )}>
                    <Zap className="w-4 h-4" />
                  </div>
                  <div>
                    {rec.priority && <div className="text-xs font-bold text-accent uppercase tracking-wider mb-1">High Priority Action</div>}
                    <p className="text-foreground text-sm md:text-base">{rec.text}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === "timeline" && (
            <div className="h-[400px] w-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={timelineData}
                  layout="vertical"
                  margin={{ top: 0, right: 30, left: 100, bottom: 0 }}
                >
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={12} tickFormatter={(val) => `${val}ms`} />
                  <YAxis dataKey="name" type="category" stroke="hsl(var(--muted-foreground))" fontSize={12} width={150} />
                  <RechartsTooltip 
                    cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px' }}
                    itemStyle={{ color: 'hsl(var(--primary))' }}
                  />
                  <Bar dataKey="duration" radius={[0, 4, 4, 0]}>
                    {timelineData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.status === 'completed' ? 'hsl(var(--primary))' : 'hsl(var(--muted))'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
