import { expect, test } from "@playwright/test";

test("semantic metric source fields render and save their DSL field_id binding", async ({ page }) => {
  const groupId = "11111111-1111-4111-8111-111111111111";
  const customersDatasetId = "22222222-2222-4222-8222-222222222222";
  const paymentsDatasetId = "33333333-3333-4333-8333-333333333333";
  const modelId = "44444444-4444-4444-8444-444444444444";
  const customersEntityId = "entity_customers";
  const paymentsEntityId = "entity_payments";
  const cityFieldId = "field_customer_city";
  const freightFieldId = "field_freight_value";
  const paymentFieldId = "field_payment_value";
  const datasets = [
    { dataset_id: customersDatasetId, name: "customers.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
    { dataset_id: paymentsDatasetId, name: "order_payments.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
  ];
  const group = {
    group_id: groupId,
    name: "Olist",
    description: "Brazilian ecommerce",
    tables: datasets.map((dataset) => ({
      dataset,
      row_count: 10,
      column_count: 2,
      columns: dataset.dataset_id === customersDatasetId
        ? ["customer_id", "customer_city"]
        : ["order_id", "freight_value", "payment_value"],
      entity_type: "unknown",
      sample_records: [],
    })),
    relationships: [],
    metadata: {},
  };
  const definition = {
    definition_schema_version: 2,
    entities: [
      {
        id: customersEntityId,
        name: "customers",
        entity_type: "dimension",
        fields: [
          { field_id: cityFieldId, source_name: "customer_city", type: "string", role: "dimension" },
        ],
      },
      {
        id: paymentsEntityId,
        name: "order_payments",
        entity_type: "fact",
        fields: [
          { field_id: freightFieldId, source_name: "freight_value", type: "number", role: "metric" },
          { field_id: paymentFieldId, source_name: "payment_value", type: "number", role: "metric" },
        ],
      },
    ],
    metrics: [
      {
        id: "metric_freight_value",
        name: "freight_value",
        aliases: [],
        format: "number",
        formula: {
          op: "sum",
          expr: { op: "field", entity_id: paymentsEntityId, field_id: freightFieldId },
        },
      },
      {
        id: "metric_missing",
        name: "missing_metric",
        aliases: [],
        format: "number",
        formula: {
          op: "sum",
          expr: { op: "field", entity_id: "missing_entity", field_id: "missing_field" },
        },
      },
    ],
    dimensions: [],
    relationships: [],
    unresolved_bindings: [],
  };
  let savedDefinition: typeof definition | null = null;
  let model = {
    model_id: modelId,
    scope_type: "dataset_group",
    scope_id: groupId,
    name: "Olist semantic model",
    version: 2,
    revision: 1,
    status: "draft",
    source: "auto",
    definition,
    validation: null,
  };

  await page.route("**/api/v1/store/datasets", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ datasets }) });
  });
  await page.route("**/api/v1/store/dataset-groups", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ groups: [group] }) });
  });
  await page.route(`**/api/v1/store/dataset-groups/${groupId}/drift`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ group_id: groupId, status: "stable", datasets: [], stale_relationship_count: 0, scanned_at: null }),
    });
  });
  await page.route("**/api/v1/store/semantic-models?**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ models: [model] }) });
  });
  await page.route(`**/api/v1/store/semantic-models/${modelId}`, async (route) => {
    if (route.request().method() !== "PUT") return route.fallback();
    const body = route.request().postDataJSON() as { definition: typeof definition };
    savedDefinition = body.definition;
    model = { ...model, revision: model.revision + 1, definition: body.definition };
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(model) });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill(`qa_semantic_binding_${Date.now()}`);
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await expect(page.getByRole("heading", { name: "工作区" })).toBeVisible();
  await page.getByRole("button", { name: /导入数据/ }).click();
  await page.getByRole("tab", { name: /关系管理/ }).click();
  await expect(page.getByRole("heading", { name: "Olist" })).toBeVisible();

  const freightSource = page.getByLabel("freight_value 来源字段");
  await expect(freightSource).toHaveValue(JSON.stringify([paymentsEntityId, freightFieldId]));
  await expect(freightSource.locator("option:checked")).toHaveText("order_payments · freight_value");
  await expect(page.getByLabel("missing_metric 来源字段")).toHaveValue("");
  await expect(page.getByLabel("missing_metric 来源字段").locator("option:checked")).toHaveText("未绑定");

  await freightSource.selectOption(JSON.stringify([paymentsEntityId, paymentFieldId]));
  await page.getByRole("button", { name: "保存" }).click();
  await expect.poll(() => savedDefinition).not.toBeNull();
  expect(savedDefinition?.metrics[0].formula.expr).toEqual({
    op: "field",
    entity_id: paymentsEntityId,
    field_id: paymentFieldId,
  });
  await expect(freightSource).toHaveValue(JSON.stringify([paymentsEntityId, paymentFieldId]));
});
