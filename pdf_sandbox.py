"""Resource-limited PDF text extraction subprocess."""

from __future__ import annotations

import io
import json
import os
import sys


PDF_MAX_RESPONSE_BYTES = max(1024, int(os.getenv("PDF_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))))
PDF_MAX_PAGES = max(1, int(os.getenv("PDF_MAX_PAGES", "200")))
PDF_MAX_EXTRACTED_CHARS = max(1000, int(os.getenv("PDF_MAX_EXTRACTED_CHARS", "1000000")))
PDF_SANDBOX_MEMORY_BYTES = max(
    128 * 1024 * 1024,
    int(os.getenv("PDF_SANDBOX_MEMORY_BYTES", str(512 * 1024 * 1024))),
)
PDF_SANDBOX_CPU_SECONDS = max(1, int(os.getenv("PDF_SANDBOX_CPU_SECONDS", "15")))


def apply_resource_limits() -> None:
    if sys.platform != "linux":
        return
    import resource

    resource.setrlimit(
        resource.RLIMIT_AS,
        (PDF_SANDBOX_MEMORY_BYTES, PDF_SANDBOX_MEMORY_BYTES),
    )
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (PDF_SANDBOX_CPU_SECONDS, PDF_SANDBOX_CPU_SECONDS + 1),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def extract_pdf(raw_body: bytes) -> dict[str, object]:
    if len(raw_body) > PDF_MAX_RESPONSE_BYTES:
        return {
            "content": "",
            "title": None,
            "error": f"PDF exceeds the {PDF_MAX_RESPONSE_BYTES}-byte limit",
        }

    try:
        from pypdf import PdfReader
    except ImportError:
        return {
            "content": "",
            "title": None,
            "error": "Direct PDF extraction requires the optional 'pypdf' package",
        }

    try:
        reader = PdfReader(io.BytesIO(raw_body))
        page_count = len(reader.pages)
        if page_count > PDF_MAX_PAGES:
            return {
                "content": "",
                "title": None,
                "error": f"PDF exceeds the {PDF_MAX_PAGES}-page limit",
            }

        parts: list[str] = []
        remaining = PDF_MAX_EXTRACTED_CHARS
        truncated = False
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            if len(page_text) > remaining:
                parts.append(page_text[:remaining])
                truncated = True
                remaining = 0
                break
            parts.append(page_text)
            remaining -= len(page_text)
            if remaining <= 0:
                truncated = True
                break

        metadata = reader.metadata
        title = str(metadata.title).strip()[:2000] if metadata and metadata.title else None
        return {
            "content": "\n\n".join(parts).strip(),
            "title": title,
            "error": (
                f"PDF extracted text was truncated at {PDF_MAX_EXTRACTED_CHARS} characters"
                if truncated
                else None
            ),
        }
    except Exception as exc:
        return {
            "content": "",
            "title": None,
            "error": f"PDF extraction failed: {type(exc).__name__}",
        }


def main() -> int:
    apply_resource_limits()
    raw_body = sys.stdin.buffer.read(PDF_MAX_RESPONSE_BYTES + 1)
    result = extract_pdf(raw_body)
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
