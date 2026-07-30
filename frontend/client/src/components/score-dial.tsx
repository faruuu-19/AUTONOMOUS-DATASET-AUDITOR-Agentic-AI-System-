import { motion } from "framer-motion";

export function ScoreDial({ score, verdict }: { score: number, verdict: string }) {
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;
  
  let colorClass = "text-success";
  let glowClass = "glow-teal";
  
  if (score < 50) {
    colorClass = "text-destructive";
    glowClass = "glow-destructive";
  } else if (score < 80) {
    colorClass = "text-warning";
    glowClass = "glow-warning";
  }

  return (
    <div className="relative flex flex-col items-center justify-center">
      <svg className="w-48 h-48 transform -rotate-90">
        {/* Background Circle */}
        <circle
          cx="96"
          cy="96"
          r={radius}
          className="stroke-muted fill-none"
          strokeWidth="8"
        />
        {/* Progress Circle */}
        <motion.circle
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          cx="96"
          cy="96"
          r={radius}
          className={`fill-none ${colorClass}`}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center text-center">
        <span className={`text-4xl font-display font-bold ${colorClass} ${glowClass} glow-text`}>
          {score}
        </span>
        <span className="text-xs uppercase tracking-widest text-muted-foreground font-semibold mt-1">
          Score
        </span>
      </div>
    </div>
  );
}
