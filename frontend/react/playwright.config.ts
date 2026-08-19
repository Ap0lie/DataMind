import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

const backendPort = process.env.DATAMIND_E2E_BACKEND_PORT ?? "18110";
const frontendPort = process.env.DATAMIND_E2E_FRONTEND_PORT ?? "15173";
const backendURL = `http://127.0.0.1:${backendPort}`;
const managedFrontendURL = `http://127.0.0.1:${frontendPort}`;
const externalFrontendURL = process.env.DATAMIND_FRONTEND_URL;
const reuseExistingServer = process.env.DATAMIND_E2E_REUSE_SERVER === "true";
const browserChannel = process.env.DATAMIND_E2E_BROWSER_CHANNEL;
const browserChannelOptions = browserChannel ? { channel: browserChannel } : {};
const projectPython = path.resolve(
  "../..",
  process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python",
);
const projectPythonAvailable =
  fs.existsSync(projectPython) &&
  spawnSync(projectPython, ["--version"], { stdio: "ignore" }).status === 0;
const systemPythonAvailable =
  spawnSync("python", ["--version"], { stdio: "ignore" }).status === 0;
const pythonCommand = projectPythonAvailable ? `"${projectPython}"` : "python";
const dockerBackendCommand = [
  "docker run --rm --init",
  `-p 127.0.0.1:${backendPort}:8000`,
  "-e DATAMIND_DATABASE_URL=",
  "-e DATAMIND_DATASET_STORE_PATH=/tmp/datamind-e2e/datasets",
  "-e DATAMIND_AUTH_MODE=session",
  "-e DATAMIND_SESSION_COOKIE_SECURE=false",
  "-e DATAMIND_EXECUTION_BACKEND=local",
  "-e DATAMIND_DEFAULT_LLM_PROVIDER=mock",
  "-e DATAMIND_CLEANING_LLM_PROVIDER=mock",
  "-e DATAMIND_PLANNER_LLM_PROVIDER=mock",
  "-e DATAMIND_SQL_LLM_PROVIDER=mock",
  "-e DATAMIND_PYTHON_LLM_PROVIDER=mock",
  "-e DATAMIND_REPORT_LLM_PROVIDER=mock",
  "-e DATAMIND_REVIEW_LLM_PROVIDER=mock",
  "-e DATAMIND_AGENT_LOOP_PROVIDER=mock",
  "-e DATAMIND_ASSISTANT_ENABLED=true",
  "-e DATAMIND_ASSISTANT_LLM_PROVIDER=mock",
  "-e DATAMIND_ASSISTANT_LLM_MODEL=mock-assistant",
  "-e DATAMIND_SEMANTIC_EMBEDDING_ENABLED=false",
  `-e DATAMIND_CORS_ORIGINS=${managedFrontendURL}`,
  "datamind-app:latest",
  `uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --log-level warning`,
].join(" ");
const defaultBackendCommand =
  projectPythonAvailable || systemPythonAvailable
    ? `${pythonCommand} -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port ${backendPort} --log-level warning`
    : dockerBackendCommand;
const isolatedDataRoot = path.join(
  os.tmpdir(),
  "datamind-playwright",
  process.env.DATAMIND_E2E_RUN_ID ?? String(process.pid),
);

export default defineConfig({
  testDir: "./e2e",
  outputDir: path.resolve("test-results"),
  reporter: process.env.CI
    ? [["line"], ["html", { outputFolder: "playwright-report", open: "never" }]]
    : "line",
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : undefined,
  timeout: 60_000,
  webServer: externalFrontendURL ? [] : [
    {
      command:
        process.env.DATAMIND_E2E_BACKEND_COMMAND ??
        defaultBackendCommand,
      cwd: path.resolve("../.."),
      url: `${backendURL}/api/v1/health/ready`,
      reuseExistingServer,
      timeout: 120_000,
      env: {
        ...process.env,
        DATAMIND_DATABASE_URL: "",
        DATAMIND_DATASET_STORE_PATH: path.join(isolatedDataRoot, "datasets"),
        DATAMIND_AUTH_MODE: "session",
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
        DATAMIND_CORS_ORIGINS: managedFrontendURL,
      },
    },
    {
      command: `npm run dev -- --port ${frontendPort} --strictPort`,
      url: managedFrontendURL,
      reuseExistingServer,
      timeout: 120_000,
      env: {
        ...process.env,
        VITE_DATAMIND_API_BASE_URL: `${backendURL}/api/v1`,
      },
    },
  ],
  use: {
    baseURL: externalFrontendURL ?? managedFrontendURL,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop-chromium",
      use: { ...devices["Desktop Chrome"], ...browserChannelOptions },
    },
    {
      name: "mobile-chromium",
      use: { ...devices["Pixel 7"], ...browserChannelOptions },
    },
  ],
});
