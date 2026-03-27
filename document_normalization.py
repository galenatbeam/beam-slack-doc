"""Helpers for normalizing MCP payloads into searchable document models."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from search_models import Citation, DocumentResult

TITLE_KEYS = ("title", "name", "pageTitle")
URL_KEYS = ("url", "webLink", "link", "pageUrl")
EXCERPT_KEYS = ("excerpt", "snippet", "summary", "text")
ID_KEYS = ("id", "pageId", "contentId", "entityId")
ARI_KEYS = ("ari", "resourceAri", "resourceARI", "resourceId")
CONTENT_KEYS = ("body", "content", "markdown", "value")


def normalize_documents(payload: Any, source: str, *, max_results: int) -> list[DocumentResult]:
    extracted = extract_payload(payload)
    candidates = list(iter_candidate_dicts(extracted))
    if isinstance(extracted, dict) and not candidates:
        candidates = [extracted]

    documents: list[DocumentResult] = []
    seen: set[str] = set()
    for candidate in candidates:
        identifier = first_string(candidate, ID_KEYS)
        ari = first_string(candidate, ARI_KEYS)
        title = first_string(candidate, TITLE_KEYS) or identifier or ari or first_string(candidate, EXCERPT_KEYS)
        url = first_string(candidate, URL_KEYS)
        excerpt = first_string(candidate, EXCERPT_KEYS)
        content = extract_text_blob(candidate)
        if content == excerpt:
            content = None

        if not title:
            continue

        dedupe_key = url or ari or identifier or title
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        documents.append(
            DocumentResult(
                title=title,
                url=url,
                excerpt=excerpt,
                identifier=identifier,
                ari=ari,
                content=content,
                source=source,
            )
        )

    return documents[:max_results]


def normalize_citations(
    citations: Sequence[Citation],
    documents: Sequence[DocumentResult],
) -> list[Citation]:
    by_title = {document.title: document for document in documents}
    normalized: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        document = by_title.get(citation.title)
        url = citation.url or (document.url if document else None)
        identifier = citation.identifier or (document.identifier if document else None)
        ari = citation.ari or (document.ari if document else None)
        reference = url or ari or identifier
        if not citation.title or not reference:
            continue
        key = (citation.title, reference)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            Citation(
                title=citation.title,
                url=url,
                excerpt=citation.excerpt or (document.excerpt if document else None),
                identifier=identifier,
                ari=ari,
            )
        )
    return normalized


def fallback_citations(documents: Sequence[DocumentResult], *, limit: int) -> list[Citation]:
    citations: list[Citation] = []
    for document in documents:
        if document.url or document.ari or document.identifier:
            citations.append(
                Citation(
                    title=document.title,
                    url=document.url,
                    excerpt=document.excerpt,
                    identifier=document.identifier,
                    ari=document.ari,
                )
            )
    return citations[:limit]


def merge_document(
    original: DocumentResult,
    fetched_documents: Sequence[DocumentResult],
    fallback_content: str | None,
) -> DocumentResult:
    if not fetched_documents:
        if fallback_content:
            original.content = fallback_content
        return original

    fetched = fetched_documents[0]
    return DocumentResult(
        title=fetched.title or original.title,
        url=fetched.url or original.url,
        excerpt=fetched.excerpt or original.excerpt,
        identifier=fetched.identifier or original.identifier,
        ari=fetched.ari or original.ari,
        content=fetched.content or fallback_content or original.content,
        source=fetched.source,
    )


def extract_payload(payload: Any) -> Any:
    if hasattr(payload, "structuredContent") and payload.structuredContent is not None:
        return payload.structuredContent

    if hasattr(payload, "content"):
        text_fragments: list[str] = []
        for item in payload.content:
            if getattr(item, "type", None) == "text":
                text_fragments.append(item.text)
        if len(text_fragments) == 1:
            return maybe_parse_json(text_fragments[0])
        if text_fragments:
            return [maybe_parse_json(fragment) for fragment in text_fragments]

    return payload


def iter_candidate_dicts(value: Any):
    if isinstance(value, dict):
        if any(key in value for key in TITLE_KEYS + URL_KEYS + ID_KEYS + ARI_KEYS):
            yield value
        for nested in value.values():
            yield from iter_candidate_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from iter_candidate_dicts(item)


def iter_nested_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from iter_nested_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from iter_nested_dicts(item)


def extract_text_blob(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in CONTENT_KEYS:
            nested = value.get(key)
            text = extract_text_blob(nested)
            if text:
                return text
        return None
    if isinstance(value, list):
        parts = [part for item in value if (part := extract_text_blob(item))]
        return "\n\n".join(parts) if parts else None
    return None


def first_string(mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def maybe_parse_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return stripped