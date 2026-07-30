import { useQuery, useMutation } from "@tanstack/react-query";
import { api, buildUrl } from "@shared/routes";
import { parseWithLogging } from "../lib/utils";

// Start new audit
export function useStartAudit() {
  return useMutation({
    mutationFn: async (data: { file: File; targetColumn: string }) => {
      const formData = new FormData();
      formData.append("file", data.file);
      formData.append("targetColumn", data.targetColumn);

      const res = await fetch(api.audit.start.path, {
        method: api.audit.start.method,
        body: formData,
        credentials: "include",
      });
      
      if (!res.ok) {
        let message = "Failed to start audit";
        try {
          const body = await res.json();
          if (body?.message) message = body.message;
        } catch (_err) {
          // Ignore JSON parse errors and use default message.
        }
        throw new Error(message);
      }
      
      const json = await res.json();
      return parseWithLogging(api.audit.start.responses[200], json, "audit.start");
    }
  });
}

// Poll audit status
export function useAuditStatus(id: string) {
  return useQuery({
    queryKey: [api.audit.status.path, id],
    queryFn: async () => {
      const url = buildUrl(api.audit.status.path, { id });
      const res = await fetch(url, { credentials: "include" });
      if (res.status === 404) throw new Error("Audit not found");
      if (!res.ok) throw new Error("Failed to fetch status");
      
      const json = await res.json();
      return parseWithLogging(api.audit.status.responses[200], json, "audit.status");
    },
    // Poll every 1s if not complete
    refetchInterval: (query) => {
      const data = query.state.data as any;
      return data?.isComplete ? false : 1000;
    },
    enabled: !!id,
  });
}

// Fetch final report
export function useAuditReport(id: string) {
  return useQuery({
    queryKey: [api.audit.report.path, id],
    queryFn: async () => {
      const url = buildUrl(api.audit.report.path, { id });
      const res = await fetch(url, { credentials: "include" });

      if (!res.ok) {
        // Surface the backend's reason (e.g. "Audit failed: <error>") instead of
        // collapsing every failure into a generic not-found message.
        let message = res.status === 404 ? "Report not found" : "Failed to fetch report";
        try {
          const body = await res.json();
          if (body?.message) message = body.message;
        } catch (_err) {
          // Non-JSON error body; keep the default message.
        }
        throw new Error(message);
      }
      
      const json = await res.json();
      return parseWithLogging(api.audit.report.responses[200], json, "audit.report");
    },
    enabled: !!id,
  });
}
