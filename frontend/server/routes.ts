import type { Express } from "express";
import type { Server } from "http";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { z } from "zod";
import { auditStartRequestSchema } from "@shared/schema";

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {

  app.post(api.audit.start.path, async (req, res) => {
    try {
      const input = auditStartRequestSchema.parse(req.body);
      const auditId = await storage.startAudit(input);
      
      res.json({ id: auditId, message: "Audit started successfully" });
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({
          message: err.errors[0].message,
          field: err.errors[0].path.join('.'),
        });
      }
      res.status(500).json({ message: "Internal server error" });
    }
  });

  app.get(api.audit.status.path, async (req, res) => {
    const id = req.params.id;
    const status = await storage.getAuditStatus(id);
    
    if (!status) {
      return res.status(404).json({ message: "Audit not found" });
    }
    
    res.json(status);
  });

  app.get(api.audit.report.path, async (req, res) => {
    const id = req.params.id;
    const report = await storage.getAuditReport(id);
    
    if (!report) {
      return res.status(404).json({ message: "Report not found or not yet ready" });
    }
    
    res.json(report);
  });

  return httpServer;
}
