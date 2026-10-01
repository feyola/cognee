import { normalizePipelineRuns } from "../normalizePipelineRuns";

const baseRun = {
  id: "legacy-run",
  dataset_id: "dataset-1",
  dataset_name: "Dataset 1",
  owner_email: null,
  created_at: null,
  pipeline_run_id: "pipeline-run-1",
};

describe("normalizePipelineRuns", () => {
  it("normalizes nullable legacy pipeline fields for dashboard rendering", () => {
    expect(
      normalizePipelineRuns([{ ...baseRun, pipeline_name: null, status: null }]),
    ).toEqual([{ ...baseRun, pipeline_name: "", status: "" }]);
  });

  it("preserves populated pipeline fields", () => {
    expect(
      normalizePipelineRuns([
        {
          ...baseRun,
          pipeline_name: "cognify_pipeline",
          status: "DATASET_PROCESSING_COMPLETED",
        },
      ]),
    ).toEqual([
      {
        ...baseRun,
        pipeline_name: "cognify_pipeline",
        status: "DATASET_PROCESSING_COMPLETED",
      },
    ]);
  });

  it("returns an empty collection for malformed top-level responses", () => {
    expect(normalizePipelineRuns({ detail: "unexpected response" })).toEqual([]);
  });
});
