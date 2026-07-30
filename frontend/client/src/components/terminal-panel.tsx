import { useEffect, useRef } from "react";
import { Terminal } from "lucide-react";
import { motion } from "framer-motion";

export function TerminalPanel({ messages, status }: { messages: string[], status: string }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="glass-panel rounded-xl overflow-hidden relative border border-primary/20 h-full flex flex-col">
      {status === "running" && <div className="scanline" />}
      
      <div className="bg-black/40 border-b border-white/5 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2 text-primary">
          <Terminal className="h-4 w-4" />
          <span className="font-mono text-xs uppercase tracking-wider font-semibold">Execution Logs</span>
        </div>
        <div className="flex gap-2">
          <div className="w-3 h-3 rounded-full bg-destructive/80" />
          <div className="w-3 h-3 rounded-full bg-warning/80" />
          <div className="w-3 h-3 rounded-full bg-success/80" />
        </div>
      </div>
      
      <div 
        ref={scrollRef}
        className="p-4 overflow-y-auto font-mono text-xs sm:text-sm text-accent/90 h-full flex-1 min-h-[300px]"
      >
        {messages.length === 0 ? (
          <span className="text-muted-foreground">Waiting for execution to begin...</span>
        ) : (
          messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="mb-1 pb-1 border-b border-white/5 last:border-0"
            >
              <span className="text-muted-foreground mr-2">{'>'}</span> 
              {msg}
            </motion.div>
          ))
        )}
        {status === "running" && (
          <motion.div 
            animate={{ opacity: [1, 0] }} 
            transition={{ repeat: Infinity, duration: 0.8 }}
            className="mt-2 inline-block w-2 h-4 bg-primary"
          />
        )}
      </div>
    </div>
  );
}
