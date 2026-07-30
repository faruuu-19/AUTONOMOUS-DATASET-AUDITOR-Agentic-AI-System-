import { useLocation, useParams } from "wouter";
import { Layout } from "../components/layout";
import { useAuditStatus } from "../hooks/use-audit";
import { TerminalPanel } from "../components/terminal-panel";
import { motion } from "framer-motion";
import { CheckCircle, CircleDashed, Loader2, ArrowRight, AlertTriangle } from "lucide-react";
import { cn } from "../lib/utils";

export function AuditRun() {
  const params = useParams();
  const [, setLocation] = useLocation();
  const { data: status, isLoading, isError } = useAuditStatus(params.id || "");

  if (isLoading) {
    return (
      <Layout>
        <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
          <Loader2 className="w-12 h-12 text-primary animate-spin" />
          <div className="text-xl font-display font-semibold animate-pulse text-primary glow-text">Initializing Agent Swarm...</div>
        </div>
      </Layout>
    );
  }

  if (isError || !status) {
    return (
      <Layout>
        <div className="text-center text-destructive mt-20">Error loading audit status.</div>
      </Layout>
    );
  }

  const activeStage = status.stages.find(s => s.status === "running") || status.stages[status.stages.length - 1];
  const messages = activeStage?.liveMessages || [];

  return (
    <Layout>
      <div className="relative">
        {/* Complete Overlay */}
        {status.isComplete && (
          <motion.div 
            initial={{ opacity: 0, backdropFilter: "blur(0px)" }}
            animate={{ opacity: 1, backdropFilter: "blur(12px)" }}
            className="absolute inset-[-2rem] z-50 flex items-center justify-center bg-background/60 rounded-3xl"
          >
            <div className="glass-panel p-10 rounded-2xl flex flex-col items-center text-center max-w-lg border-primary/30 glow-cyan">
              <div className="w-20 h-20 bg-success/20 rounded-full flex items-center justify-center mb-6 glow-teal">
                <CheckCircle className="w-10 h-10 text-success" />
              </div>
              <h2 className="text-3xl font-display font-bold mb-4">Audit Complete</h2>
              <p className="text-muted-foreground mb-8">
                The agent swarm has finished analyzing your dataset. Insights, potential leaks, and fixes are ready.
              </p>
              <button
                onClick={() => setLocation(`/audit/${status.id}/report`)}
                className="w-full px-8 py-4 rounded-xl bg-primary text-primary-foreground font-bold text-lg flex items-center justify-center gap-2 hover:bg-primary/90 glow-cyan transition-all"
              >
                View Detailed Report
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        )}

        <div className="flex flex-col md:flex-row justify-between items-start mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-display font-bold mb-2">Live Execution Engine</h1>
            <p className="text-muted-foreground">Audit ID: <span className="font-mono text-xs">{status.id}</span></p>
          </div>
          <div className="glass-panel px-6 py-3 rounded-full flex items-center gap-4 border-primary/20">
            <div className="text-sm font-semibold text-muted-foreground">OVERALL PROGRESS</div>
            <div className="w-48 h-2 bg-black/50 rounded-full overflow-hidden">
              <motion.div 
                className="h-full bg-primary glow-cyan"
                initial={{ width: 0 }}
                animate={{ width: `${status.progressPercentage}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
            <div className="text-lg font-mono font-bold text-primary">{Math.round(status.progressPercentage)}%</div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-[600px]">
          {/* Pipeline Sidebar */}
          <div className="lg:col-span-4 glass-panel rounded-2xl p-6 overflow-y-auto">
            <h3 className="text-lg font-display font-bold mb-6 text-foreground">Pipeline Stages</h3>
            <div className="space-y-0">
              {status.stages.map((stage, idx) => {
                const isRunning = stage.status === "running";
                const isComplete = stage.status === "completed";
                const isSkipped = stage.status === "skipped";
                
                return (
                  <div key={stage.id} className="relative flex gap-4 pb-8 last:pb-0">
                    {/* Connection Line */}
                    {idx < status.stages.length - 1 && (
                      <div className="absolute top-8 left-3 w-px h-full bg-white/10 -translate-x-1/2 z-0" />
                    )}
                    
                    {/* Node Icon */}
                    <div className="relative z-10">
                      {isComplete ? (
                        <div className="w-6 h-6 rounded-full bg-success/20 flex items-center justify-center text-success glow-teal">
                          <CheckCircle className="w-4 h-4" />
                        </div>
                      ) : isRunning ? (
                        <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center text-primary glow-cyan">
                          <Loader2 className="w-4 h-4 animate-spin" />
                        </div>
                      ) : (
                        <div className="w-6 h-6 rounded-full bg-black flex items-center justify-center text-muted-foreground border border-white/20">
                          <CircleDashed className="w-4 h-4" />
                        </div>
                      )}
                    </div>
                    
                    {/* Content */}
                    <div className="flex-1 -mt-1">
                      <div className={cn(
                        "font-semibold text-base transition-colors",
                        isRunning ? "text-primary glow-text" : isComplete ? "text-foreground" : "text-muted-foreground"
                      )}>
                        {stage.name}
                      </div>
                      
                      {stage.criticAssessment && (
                        <div className="mt-2 p-3 bg-black/40 rounded-lg border border-warning/30 flex gap-2">
                          <AlertTriangle className="w-4 h-4 text-warning flex-shrink-0 mt-0.5" />
                          <div className="text-xs text-muted-foreground">
                            <span className="text-warning font-semibold">Critic Notice:</span> {stage.criticAssessment.message}
                          </div>
                        </div>
                      )}
                      {isSkipped && (
                        <div className="text-xs text-muted-foreground mt-1 italic">Skipped: {stage.skipReason}</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Terminal Area */}
          <div className="lg:col-span-8">
            <TerminalPanel 
              messages={messages} 
              status={activeStage?.status || "pending"} 
            />
          </div>
        </div>
      </div>
    </Layout>
  );
}
