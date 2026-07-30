import { z } from "zod";
import { 
  auditStartRequestSchema, 
  auditStartResponseSchema, 
  auditStatusSchema, 
  auditReportSchema 
} from "./schema";

export const errorSchemas = {
  validation: z.object({
    message: z.string(),
    field: z.string().optional(),
  }),
  notFound: z.object({
    message: z.string(),
  }),
  internal: z.object({
    message: z.string(),
  }),
};

export const api = {
  audit: {
    start: {
      method: "POST" as const,
      path: "/api/audit/start" as const,
      input: auditStartRequestSchema,
      responses: {
        200: auditStartResponseSchema,
        400: errorSchemas.validation,
      },
    },
    status: {
      method: "GET" as const,
      path: "/api/audit/:id/status" as const,
      responses: {
        200: auditStatusSchema,
        404: errorSchemas.notFound,
      },
    },
    report: {
      method: "GET" as const,
      path: "/api/audit/:id/report" as const,
      responses: {
        200: auditReportSchema,
        404: errorSchemas.notFound,
      },
    },
  },
};

export function buildUrl(path: string, params?: Record<string, string | number>): string {
  let url = path;
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (url.includes(`:${key}`)) {
        url = url.replace(`:${key}`, String(value));
      }
    });
  }
  return url;
}
