import os from "node:os";
import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  outputDir: path.join(os.tmpdir(), "datamind-playwright-results"),
  reporter: "line",
  timeout: 60_000,
  webServer: [
    {
      command:
        process.env.DATAMIND_E2E_BACKEND_COMMAND ??
        "python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8010 --log-level warning",
      cwd: path.resolve("../.."),
      url: "http://127.0.0.1:8010/api/v1/health/ready",
      reuseExistingServer: true,
      timeout: 120_000,
      env: {
        ...process.env,
        DATAMIND_DATABASE_URL: "",
        DATAMIND_DATASET_STORE_PATH: path.join(os.tmpdir(), "datamind-playwright", "datasets"),
        DATAMIND_AUTH_MODE: "legacy",
        DATAMIND_SESSION_COOKIE_SECURE: "false",
        DATAMIND_EXECUTION_BACKEND: "local",
        DATAMIND_DEFAULT_LLM_PROVIDER: "mock",
        DATAMIND_CLEANING_LLM_PROVIDER: "mock",
        DATAMIND_PLANNER_LLM_PROVIDER: "mock",
        DATAMIND_SQL_LLM_PROVIDER: "mock",
        DATAMIND_PYTHON_LLM_PROVIDER: "mock",
        DATAMIND_REPORT_LLM_PROVIDER: "mock",
        DATAMIND_REVIEW_LLM_PROVIDER: "mock",
        DATAMIND_AGENT_LOOP_PROVIDER: "mock",
        DATAMIND_ASSISTANT_ENABLED: "true",
        DATAMIND_ASSISTANT_LLM_PROVIDER: "mock",
        DATAMIND_ASSISTANT_LLM_MODEL: "mock-assistant",
        DATAMIND_SEMANTIC_EMBEDDING_ENABLED: "false",
      },
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
  use: {
    baseURL: process.env.DATAMIND_FRONTEND_URL ?? "http://127.0.0.1:5173",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop-chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chromium", use: { ...devices["Pixel 7"] } },
  ],
});
