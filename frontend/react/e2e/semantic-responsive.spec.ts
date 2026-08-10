import { expect, test } from "@playwright/test";

test("semantic model toolbar stays usable at a narrow viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });

  const ordersId = "18181818-1818-4818-8818-181818181818";
  const customersId = "19191919-1919-4919-8919-191919191919";
  const groupId = "20202020-2020-4020-8020-202020202020";
  const firstModelId = "21212121-2121-4121-8121-212121212121";
  const secondModelId = "22222222-2222-4222-8222-222222222222";
  const conversationId = "24242424-2424-4424-8424-242424242424";
  const datasets = [
    { dataset_id: ordersId, name: "orders.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
    { dataset_id: customersId, name: "customers.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
  ];
  const group = {
    group_id: groupId,
    name: "Responsive semantic model fixture with a deliberately long name",
    description: "A long description exercises the grid item's intrinsic width.",
    tables: [
      { dataset: datasets[0], row_count: 1000, column_count: 3, columns: ["order_id", "customer_id", "gross_payment_value_with_a_long_name"], entity_type: "fact", sample_records: [] },
      { dataset: datasets[1], row_count: 200, column_count: 2, columns: ["customer_id", "customer_segment_with_a_long_name"], entity_type: "dimension", sample_records: [] },
    ],
    relationships: [{
      left_dataset_id: ordersId,
      right_dataset_id: customersId,
      left_column: "customer_id_with_a_deliberately_long_relationship_label",
      right_column: "customer_id_with_a_deliberately_long_relationship_label",
      join_type: "left",
      enabled: true,
      confidence: 0.98,
      source: "rules",
      estimated_match_rate: 0.99,
      reason: "Validated relationship.",
      relationship_type: "many_to_one",
    }],
    metadata: {},
  };
  const definition = {
    entities: [{
      id: "orders",
      name: "orders.csv",
      entity_type: "fact",
      grain: "one row per order",
      fields: [{ id: "amount", name: "amount", source_name: "gross_payment_value_with_a_long_name", type: "number", role: "measure" }],
    }],
    dimensions: [],
    metrics: [{ id: "gross_payment_value", name: "Gross payment value", formula: { op: "sum", entity_id: "orders", field_id: "amount" } }],
    relationships: [],
    unresolved_bindings: [],
  };
  const firstModel = {
    model_id: firstModelId,
    scope_type: "dataset_group",
    scope_id: groupId,
    name: "Responsive semantic model",
    version: 1,
    revision: 1,
    status: "draft",
    source: "auto",
    definition,
    validation: null,
  };
  let activeModel = firstModel;
  let models = [firstModel];
  let createCalls = 0;
  let saveCalls = 0;
  let validateCalls = 0;
  let publishCalls = 0;

  await page.route("**/api/v1/**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });
  await page.route("**/api/v1/assistant/**", async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    if (route.request().method() === "GET" && pathname.endsWith("/assistant/conversations")) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          conversations: [{
            conversation_id: conversationId,
            title: "语义模型测试对话",
            scope_type: "auto",
            scope_id: null,
            summary: "",
            active_run_id: null,
            active_run_status: null,
            created_at: "2026-08-08T00:00:00Z",
            updated_at: "2026-08-08T00:00:00Z",
            last_message_at: null,
          }],
        }),
      });
      return;
    }
    if (route.request().method() === "GET" && pathname.endsWith(`/assistant/conversations/${conversationId}/messages`)) {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ messages: [] }) });
      return;
    }
    await route.fallback();
  });
  await page.route("**/api/v1/auth/login", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        user_id: "23232323-2323-4323-8323-232323232323",
        display_name: "Semantic responsive QA",
        csrf_token: "semantic-responsive-csrf",
        expires_at: "2099-08-08T00:00:00Z",
        created: true,
      }),
    });
  });
  await page.route("**/api/v1/store/datasets", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ datasets }) });
  });
  await page.route("**/api/v1/store/dataset-groups", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ groups: [group] }) });
  });
  await page.route("**/api/v1/store/dataset-groups/*/drift**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ group_id: groupId, status: "stable", datasets: [], stale_relationship_count: 0, scanned_at: "2026-08-07T00:00:00Z" }) });
  });
  await page.route("**/api/v1/store/reports**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ reports: [] }) });
  });
  await page.route("**/api/v1/analysis/jobs**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [] }) });
  });
  await page.route("**/api/v1/store/cleaning-jobs**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [] }) });
  });
  await page.route("**/api/v1/store/semantic-models**", async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (request.method() === "GET" && pathname.endsWith("/semantic-models")) {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ models }) });
      return;
    }
    if (request.method() === "POST" && pathname.endsWith("/semantic-models/drafts")) {
      createCalls += 1;
      activeModel = { ...firstModel, model_id: secondModelId, version: 2, revision: 1 };
      models = [activeModel, firstModel];
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(activeModel) });
      return;
    }
    if (request.method() === "PUT" && pathname.endsWith(`/${activeModel.model_id}`)) {
      saveCalls += 1;
      const payload = request.postDataJSON();
      activeModel = { ...activeModel, revision: activeModel.revision + 1, definition: payload.definition };
      models = [activeModel, firstModel];
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(activeModel) });
      return;
    }
    if (request.method() === "POST" && pathname.endsWith(`/${activeModel.model_id}/validate`)) {
      validateCalls += 1;
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ valid: true, errors: [], warnings: [], schema_fingerprint: "responsive-fixture" }) });
      return;
    }
    if (request.method() === "POST" && pathname.endsWith(`/${activeModel.model_id}/publish`)) {
      publishCalls += 1;
      activeModel = { ...activeModel, status: "published" };
      models = [activeModel, firstModel];
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(activeModel) });
      return;
    }
    await route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "Unexpected semantic fixture request" }) });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_semantic_responsive");
  await page.getByLabel("密码").fill("qa-reliability-password");
  const datasetsLoaded = page.waitForResponse((response) =>
    response.request().method() === "GET"
    && new URL(response.url()).pathname === "/api/v1/store/datasets",
  );
  const groupsLoaded = page.waitForResponse((response) =>
    response.request().method() === "GET"
    && new URL(response.url()).pathname === "/api/v1/store/dataset-groups",
  );
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await Promise.all([datasetsLoaded, groupsLoaded]);
  await page.getByRole("button", { name: "数据集" }).click();
  await page.getByRole("tab", { name: /关系管理/ }).click();

  const workbench = page.getByRole("heading", { name: "语义模型", exact: true }).locator("xpath=ancestor::section[1]");
  await expect(workbench).toBeVisible();
  const workbenchBox = await workbench.boundingBox();
  expect(workbenchBox).not.toBeNull();
  expect(workbenchBox!.x).toBeGreaterThanOrEqual(0);
  expect(workbenchBox!.x + workbenchBox!.width).toBeLessThanOrEqual(390);

  const toolbarNames = ["新建版本", "保存", "校验", "发布", "可视化", "高级 JSON"];
  for (const name of toolbarNames) {
    const button = workbench.getByRole("button", { name, exact: true });
    await expect(button).toBeVisible();
    const box = await button.boundingBox();
    expect(box, `${name} should have a rendered box`).not.toBeNull();
    expect(box!.x, `${name} should not be clipped on the left`).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width, `${name} should not be clipped on the right`).toBeLessThanOrEqual(390);
  }

  await workbench.getByRole("button", { name: "新建版本", exact: true }).click();
  await expect(workbench.getByRole("button", { name: "v2 · 草稿", exact: true })).toBeVisible();
  await workbench.getByRole("button", { name: "保存", exact: true }).click();
  await expect(workbench.getByText("语义草稿已保存。", { exact: true })).toBeVisible();
  await workbench.getByRole("button", { name: "校验", exact: true }).click();
  await expect(workbench.getByText("语义模型校验通过", { exact: true })).toBeVisible();
  await workbench.getByRole("button", { name: "高级 JSON", exact: true }).click();
  await expect(workbench.getByLabel("语义模型 DSL JSON")).toBeVisible();
  await workbench.getByRole("button", { name: "可视化", exact: true }).click();
  await expect(workbench.getByRole("heading", { name: /^指标(?:\s+\d+)?$/ })).toBeVisible();
  await workbench.getByRole("button", { name: "发布", exact: true }).click();
  await expect(workbench.getByText("语义模型已发布，后续 Planner 将固定引用该版本。", { exact: true })).toBeVisible();
  expect({ createCalls, saveCalls, validateCalls, publishCalls }).toEqual({ createCalls: 1, saveCalls: 3, validateCalls: 2, publishCalls: 1 });
});
