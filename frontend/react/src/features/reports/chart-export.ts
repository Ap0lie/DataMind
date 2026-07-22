export async function exportChart(
  container: HTMLDivElement | null,
  title: string,
  format: "svg" | "png",
) {
  const source = container?.querySelector("svg");
  if (!source) return;
  const svg = source.cloneNode(true) as SVGSVGElement;
  svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const viewBox = svg.viewBox.baseVal;
  const width = Math.max(viewBox.width || source.clientWidth || 760, 320);
  const height = Math.max(viewBox.height || source.clientHeight || 300, 180);
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  const content = new XMLSerializer().serializeToString(svg);
  const fileName = safeFileName(title);
  if (format === "svg") {
    downloadBlob(
      new Blob([content], { type: "image/svg+xml;charset=utf-8" }),
      `${fileName}.svg`,
    );
    return;
  }

  const objectUrl = URL.createObjectURL(
    new Blob([content], { type: "image/svg+xml;charset=utf-8" }),
  );
  try {
    const image = await loadImage(objectUrl);
    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(scale, scale);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/png", 1),
    );
    if (blob) downloadBlob(blob, `${fileName}.png`);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function loadImage(source: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("图表图片生成失败。"));
    image.src = source;
  });
}

function downloadBlob(blob: Blob, fileName: string) {
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = fileName;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(href), 0);
}

function safeFileName(value: string) {
  return value.replace(/[\\/:*?"<>|]/g, "_").trim().slice(0, 80) || "datamind-chart";
}
