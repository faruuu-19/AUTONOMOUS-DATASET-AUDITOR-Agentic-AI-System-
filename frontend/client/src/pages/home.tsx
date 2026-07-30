import { useState } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { UploadCloud, FileSpreadsheet, ArrowRight, Target, Database } from "lucide-react";
import { Layout } from "../components/layout";
import { useStartAudit } from "../hooks/use-audit";
import { cn } from "../lib/utils";

const HEADER_READ_BYTES = 1024 * 1024;

async function detectCsvColumns(file: File): Promise<string[]> {
  const chunk = await file.slice(0, HEADER_READ_BYTES).text();
  const firstLine = chunk
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line.length > 0);

  if (!firstLine) return [];

  return firstLine
    .split(",")
    .map((value) => value.trim().replace(/^"|"$/g, ""))
    .filter((value) => value.length > 0);
}

export function Home() {
  const [, setLocation] = useLocation();
  const startAudit = useStartAudit();

  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [headerError, setHeaderError] = useState<string>("");

  const onFileSelected = async (nextFile: File | null) => {
    if (!nextFile) {
      setFile(null);
      setTargetColumn("");
      setAvailableColumns([]);
      setHeaderError("");
      return;
    }

    setFile(nextFile);
    setHeaderError("");

    try {
      const columns = await detectCsvColumns(nextFile);
      setAvailableColumns(columns);

      if (columns.length === 0) {
        setTargetColumn("");
        setHeaderError("Could not detect CSV headers. Please upload a valid CSV file.");
        return;
      }

      setTargetColumn((current) => (columns.includes(current) ? current : columns[columns.length - 1]));
    } catch (_err) {
      setAvailableColumns([]);
      setTargetColumn("");
      setHeaderError("Failed to read CSV headers.");
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleStart = () => {
    if (!file || !targetColumn) return;

    startAudit.mutate(
      { file, targetColumn },
      {
        onSuccess: (data) => {
          setLocation(`/audit/${data.id}`);
        },
      }
    );
  };

  return (
    <Layout>
      <div className="max-w-4xl mx-auto mt-12 flex flex-col items-center">
        <div className="text-center mb-12">
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-5xl font-display font-bold mb-4"
          >
            Autonomous <span className="text-primary glow-text">Data Audit</span>
          </motion.h1>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Detect leakage, bias, and spurious correlations in your ML pipelines before they reach production.
          </p>
        </div>

        <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-8">
          <div
            className={cn(
              "glass-panel rounded-2xl p-8 flex flex-col items-center justify-center text-center transition-all duration-300 min-h-[320px] border-2 border-dashed relative overflow-hidden",
              isDragging ? "border-primary bg-primary/5 glow-cyan" : "border-white/10 hover:border-primary/50",
              file ? "border-solid border-accent/30 bg-accent/5" : ""
            )}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            {file ? (
              <motion.div initial={{ scale: 0.8 }} animate={{ scale: 1 }} className="flex flex-col items-center">
                <div className="w-16 h-16 rounded-full bg-accent/20 flex items-center justify-center mb-4 glow-teal">
                  <FileSpreadsheet className="w-8 h-8 text-accent" />
                </div>
                <h3 className="text-xl font-bold text-foreground mb-1">{file.name}</h3>
                <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB | CSV</p>
                <button onClick={() => onFileSelected(null)} className="mt-6 text-sm text-destructive hover:text-destructive/80">
                  Remove file
                </button>
              </motion.div>
            ) : (
              <>
                <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mb-6">
                  <UploadCloud className="w-10 h-10 text-primary" />
                </div>
                <h3 className="text-xl font-display font-bold mb-2">Drop your dataset here</h3>
                <p className="text-muted-foreground mb-6">Supports CSV files</p>
                <label className="px-6 py-3 rounded-lg bg-primary/10 text-primary font-semibold hover:bg-primary/20 border border-primary/20 transition-all cursor-pointer">
                  Browse Files
                  <input type="file" className="hidden" accept=".csv" onChange={(e) => onFileSelected(e.target.files?.[0] || null)} />
                </label>
              </>
            )}
          </div>

          <div
            className={cn(
              "glass-panel rounded-2xl p-8 flex flex-col justify-between transition-all duration-500",
              !file || availableColumns.length === 0 ? "opacity-50 pointer-events-none grayscale" : ""
            )}
          >
            <div>
              <div className="flex items-center gap-3 mb-6">
                <Target className="text-secondary w-6 h-6" />
                <h3 className="text-xl font-display font-bold">Configure Audit</h3>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">Target Column (Prediction)</label>
                  <select
                    className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-3 text-foreground focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary appearance-none font-mono text-sm"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                  >
                    {availableColumns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="p-4 rounded-lg bg-black/30 border border-white/5 flex gap-4 mt-4">
                  <Database className="text-muted-foreground w-5 h-5 flex-shrink-0" />
                  <div className="text-sm">
                    <div className="text-foreground font-semibold mb-1">Dataset Profile</div>
                    <div className="text-muted-foreground font-mono">{availableColumns.length} detected columns</div>
                    <div className="text-muted-foreground font-mono mt-1">
                      File size: {((file?.size ?? 0) / 1024 / 1024).toFixed(2)} MB
                    </div>
                  </div>
                </div>

                {headerError && <div className="text-sm text-destructive">{headerError}</div>}
              </div>
            </div>

            <button
              onClick={handleStart}
              disabled={startAudit.isPending || !file || !targetColumn || availableColumns.length === 0}
              className="mt-8 w-full group relative px-6 py-4 rounded-xl font-bold text-lg text-primary-foreground bg-primary overflow-hidden transition-all hover:scale-[1.02] active:scale-[0.98] glow-cyan flex items-center justify-center gap-2 disabled:opacity-70 disabled:pointer-events-none"
            >
              <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
              {startAudit.isPending ? "Initializing Engine..." : "Initiate Audit"}
              {!startAudit.isPending && <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />}
            </button>
            {startAudit.isError && (
              <div className="mt-3 text-sm text-destructive">{(startAudit.error as Error).message}</div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}
