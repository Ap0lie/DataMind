import { expect, test } from "@playwright/test";

test("Kimi saves and manages an explicit long-term memory", async ({ page }, testInfo) => {
  const preference = "请记住，以后默认用中文并保持报告简洁。";

  await page.goto("/");
  await page.getByLabel("用户名").fill(`qa_memory_${testInfo.project.name.replace(/\W/g, "_")}_${Date.now()}`);
  await page.getByLabel("密码").fill("qa-memory-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await expect(page.getByRole("heading", { name: "工作区" })).toBeVisible();

  await page.getByRole("button", { name: "Kimi" }).click();
  const composer = page.getByPlaceholder("向 Kimi 询问你的数据、分析结果或报告...");
  await composer.fill(preference);
  await page.getByRole("button", { name: "发送" }).click();
  await expect(page.getByText("已记住这项偏好。")).toBeVisible({ timeout: 60_000 });

  await composer.fill("请按我保存的偏好说明报告语言和风格。");
  await page.getByRole("button", { name: "发送" }).click();
  const recalled = page.getByText(/使用了 \d+ 条记忆/).last();
  await expect(recalled).toBeVisible({ timeout: 60_000 });
  await recalled.click();
  const helpful = page.getByRole("button", { name: "有用", exact: true }).last();
  await helpful.click();
  await expect(helpful).toHaveClass(/active/);

  await page.getByRole("button", { name: /打开 Kimi 工作台/ }).click();
  const workbench = page.getByRole("complementary", { name: "Kimi 权限与操作" });
  await workbench.getByRole("button", { name: "记忆", exact: true }).click();
  const memorySwitch = workbench.getByRole("switch");
  await expect(memorySwitch).toHaveAttribute("aria-checked", "true");
  await memorySwitch.click();
  await expect(memorySwitch).toHaveAttribute("aria-checked", "false");
  await expect(workbench.getByText("已停止长期记忆的读取与写入；对话摘要仍保留")).toBeVisible();
  await memorySwitch.click();
  await expect(memorySwitch).toHaveAttribute("aria-checked", "true");

  await workbench.getByRole("button", { name: "分析经验", exact: true }).click();
  await expect(workbench.getByText("还没有通过统计审查并可复用的分析经验。")).toBeVisible();
  await workbench.getByRole("button", { name: "质量", exact: true }).click();
  await expect(workbench.getByText("记忆有效性")).toBeVisible();
  await workbench.getByRole("button", { name: "版本历史", exact: true }).click();
  await expect(workbench.getByText("尚无被替代、失效、休眠或回收的历史版本。")).toBeVisible();
  await workbench.getByRole("button", { name: "用户记忆", exact: true }).click();
  const memory = workbench.locator(".assistant-memory-list article").filter({ hasText: preference });
  await expect(memory).toBeVisible();
  await memory.getByRole("button", { name: "固定", exact: true }).click();
  await expect(memory.getByLabel("已固定")).toBeVisible();

  await memory.getByRole("button", { name: "回收", exact: true }).click();
  await workbench.getByLabel("记忆状态").selectOption("recycled");
  const recycled = workbench.locator(".assistant-memory-list article").filter({ hasText: preference });
  await expect(recycled).toBeVisible();
  await recycled.getByRole("button", { name: "恢复", exact: true }).click();
  await workbench.getByLabel("记忆状态").selectOption("active");
  await expect(workbench.locator(".assistant-memory-list article").filter({ hasText: preference })).toBeVisible();
});
