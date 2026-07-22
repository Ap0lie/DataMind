import { useEffect, useState } from "react";
import { ImageOff, Loader2 } from "lucide-react";
import { apiFetch } from "../../api-client";
import type { AssistantAttachment } from "./types";

export function AssistantAttachmentImage({ attachment }: { attachment: AssistantAttachment }) {
  const [source, setSource] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | null = null;
    setSource(null);
    setFailed(false);
    void apiFetch(attachment.content_url, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Image request failed: ${response.status}`);
        return response.blob();
      })
      .then((blob) => {
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setSource(objectUrl);
      })
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true);
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [attachment.attachment_id, attachment.content_url]);

  if (failed) {
    return <span className="assistant-message-image-state error"><ImageOff size={18} />图片读取失败</span>;
  }
  if (!source) {
    return <span className="assistant-message-image-state"><Loader2 className="animate-spin" size={18} />正在读取图片</span>;
  }
  return <img src={source} alt={attachment.file_name} />;
}
