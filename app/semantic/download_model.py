from __future__ import annotations

import argparse
from pathlib import Path

from app.core.settings import get_settings

_REQUIRED_MODEL_FILES = ("config.json", "tokenizer.json")
_MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def missing_model_files(output: Path) -> tuple[str, ...]:
    missing = [name for name in _REQUIRED_MODEL_FILES if not (output / name).is_file()]
    if not any((output / name).is_file() for name in _MODEL_WEIGHT_FILES):
        missing.append("model.safetensors|pytorch_model.bin")
    return tuple(missing)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Download or verify the pinned DataMind semantic embedding model.")
    parser.add_argument("--model", default=settings.semantic_embedding_model)
    parser.add_argument("--revision", default=settings.semantic_embedding_revision)
    parser.add_argument("--output", type=Path, default=Path(settings.semantic_embedding_model_path))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        missing = missing_model_files(args.output)
        if missing:
            raise SystemExit("Model verification failed; missing: " + ", ".join(missing))
        print(f"verified {args.output} revision={args.revision}")
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=args.model, revision=args.revision, local_dir=args.output)
    print(f"downloaded {args.model}@{args.revision} to {args.output}")


if __name__ == "__main__":
    main()
