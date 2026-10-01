import type { PipelineRun } from "@/ui/elements/AgentActivityTerminal";

type PipelineRunResponse = Omit<PipelineRun, "pipeline_name" | "status"> & {
  pipeline_name?: string | null;
  status?: string | null;
};

/**
 * Convert the activity endpoint's nullable legacy fields into the dashboard's
 * string-based view model. Older PipelineRun rows can have neither a pipeline
 * name nor a status, and rendering those values as strings would otherwise
 * crash the authenticated dashboard.
 */
export function normalizePipelineRuns(value: unknown): PipelineRun[] {
  if (!Array.isArray(value)) return [];

  return value
    .filter((row): row is PipelineRunResponse => typeof row === "object" && row !== null)
    .map((row) => ({
      ...row,
      pipeline_name: typeof row.pipeline_name === "string" ? row.pipeline_name : "",
      status: typeof row.status === "string" ? row.status : "",
    }));
}
