"""Patent-domain tools for a reproducible demo.

Local patent documents are explicitly fictional training records.  Claim comparison
is a lexical research aid; it cannot determine patent novelty, infringement or validity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
from functools import lru_cache

from .retrieval import KnowledgeIndex
from .bm25 import BM25Index, tokenize

_CLAIM_START = re.compile(r"^\s*(\d{1,3})\s*[.)]\s*(\S.*)$")
_CLAIM_DEPENDS = re.compile(r"\bclaims?\s+(\d{1,3})(?:\s*(?:,|or|and)\s*(\d{1,3}))?\b", flags=re.I)
# Only the dependent-claim preamble is examined; this is a heuristic, NOT a legal parser.
_CLAIM_DEPENDENCY_SEQUENCE = re.compile(
    r"\bclaims?\s+(\d{1,3})(?:(?:\s*(?:,|and|or|to|through|-)\s*\d{1,3})*)",
    flags=re.I,
)
_CLAIM_DEPENDENCY_TOKENS = re.compile(r"\d{1,3}|,|\b(?:and|or|to|through)\b|-", flags=re.I)

_TOKENS = re.compile(r"[a-z][a-z0-9-]{2,}", flags=re.I)
_QUERY_BOILERPLATE = set("""a an and are as based by describes described filing for from in is of on or patent patents
patented patenting what which who where when publication published application disclosed disclose describes
claims claim covering covers cover uses using use that the this their to through with does an one any more
was were can did type such following discusses introduces method invention technology records
""".split())
_STOP = set("a an the is are was were being to of in on for from by or and with as said said an wherein comprising including according claim claims method apparatus system that which having said thereupon".split())


@dataclass(frozen=True)
class Claim:
    number: int
    text: str
    depends_on: list[int]
    independent: bool



def _dependencies_from_preamble(head: str) -> list[int]:
    """Parse common dependent-claim ranges and lists: 'claims 1 to 3', 'claims 1, 2 or 3'.

    A range is expanded inclusively. Independent claims may mention other
    claims in explanatory text, so callers should regard output as heuristic.
    """
    dependencies: list[int] = []
    for match in _CLAIM_DEPENDENCY_SEQUENCE.finditer(head):
        tokens = [m.group(0).lower() for m in _CLAIM_DEPENDENCY_TOKENS.finditer(match.group(0))]
        numbers: list[int] = []
        if not tokens:
            continue
        numbers.append(int(tokens[0]))
        position = 1
        while position + 1 < len(tokens):
            join, next_token = tokens[position], tokens[position + 1]
            if not next_token.isdigit():
                break
            next_number = int(next_token)
            if join in {'to', 'through', '-'}:
                if not (numbers[-1] < next_number <= 100):
                    raise ValueError('Invalid or descending claim dependency range')
                numbers.extend(range(numbers[-1] + 1, next_number + 1))
            elif join in {',', 'and', 'or'}:
                numbers.append(next_number)
            else:
                break
            position += 2
        dependencies.extend(n for n in numbers if n not in dependencies)
    return dependencies

def parse_claims(raw: str) -> dict:
    """Parse line-numbered English claims; reject missing/duplicate refs.

    Supports claim headers '1. ...' and '2) ...' and the common 'of claim 1'
    pattern, not all jurisdiction/format-specific claim styles.
    """
    if not raw.strip() or len(raw) > 60_000:
        raise ValueError("claims text must be 1 to 60000 characters")
    sections: list[tuple[int, list[str]]] = []
    for line in raw.splitlines():
        match = _CLAIM_START.match(line)
        if match:
            sections.append((int(match.group(1)), [match.group(2).strip()]))
        elif sections and line.strip():
            sections[-1][1].append(line.strip())
    if not sections:
        raise ValueError("No numbered claims found (expected '1. ...' on its own line)")
    if len(sections) > 100:
        raise ValueError("At most 100 numbered claims supported")
    ids = [i for i, _ in sections]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate claim numbers")
    claims: list[Claim] = []
    warnings: list[str] = []
    for number, lines in sections:
        text = " ".join(lines)
        # Detect dependencies only in first 200 characters to avoid references
        # discussed later in the claim being mistaken for dependencies.
        head = text[:200]
        deps = _dependencies_from_preamble(head)
        for dep in deps:
            if dep not in ids:
                warnings.append(f"claim {number} references missing claim {dep}")
            if dep >= number:
                warnings.append(f"claim {number} has non-earlier dependency {dep}; inspect manually")
        claims.append(Claim(number, text, deps, not deps))
    dependency_map = {c.number: c.depends_on for c in claims}
    return {"claims": [asdict(c) for c in claims], "claim_count": len(claims),
            "dependencies": dependency_map,
            "warnings": warnings,
            "parser_scope": "basic line-numbered English claims including common numeric lists/ranges; heuristic only"}


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _TOKENS.findall(text) if w.lower() not in _STOP}


def compare_claims(target_claims: str, reference_claims: str) -> dict:
    """Create an explainable lexical coverage chart, NOT a novelty opinion."""
    target = parse_claims(target_claims)["claims"]
    reference = parse_claims(reference_claims)["claims"]
    chart = []
    for claim in target:
        target_terms = _terms(claim["text"])
        options = []
        for prior in reference:
            ref_terms = _terms(prior["text"])
            intersection = target_terms & ref_terms
            coverage = len(intersection) / len(target_terms) if target_terms else 0.0
            options.append((coverage, prior["number"], intersection))
        best = max(options, key=lambda item: item[0])
        chart.append({
            "target_claim": claim["number"], "closest_reference_claim": best[1],
            "lexical_coverage": round(best[0], 4),
            "overlapping_terms": sorted(best[2]),
            "target_only_terms": sorted(target_terms - best[2]),
        })
    return {"claim_chart": chart,
            "metric": "lexical coverage of target claim terms by best-matching reference claim",
            "disclaimer": "Research screening aid ONLY. No legal novelty, patentability, infringement or FTO conclusion."}


@lru_cache(maxsize=8)
def _cached_bm25(index: KnowledgeIndex) -> BM25Index:
    return BM25Index(index)


def search_local_patents(index: KnowledgeIndex, query: str, k: int = 3,
                         retrieval: str = "hybrid") -> dict:
    """Search separate synthetic patent folder, with explicit provenance."""
    if not 1 <= len(query.strip()) <= 400:
        raise ValueError("query must be 1-400 characters")
    if not 1 <= k <= 10:
        raise ValueError("k must be from 1 to 10")
    # Search all indexed chunks before filtering so unrelated demo documents
    # don't crowd synthetic patent matches out of the top-k.
    if retrieval == "bm25":
        hits = _cached_bm25(index).search(query, k=len(index.chunks))
    elif retrieval in ("word", "hybrid"):
        hits = index.search(query, k=len(index.chunks), mode=retrieval)
    else:
        raise ValueError("retrieval must be bm25, word or hybrid")
    filtered = [h.to_dict() for h in hits if h.doc_id.startswith("patent_demo_")][:k]
    return {"source_type": "SYNTHETIC_DEMO_NOT_REAL_PATENTS", "retrieval": retrieval, "count": len(filtered),
            "results": filtered}


def search_curated_patent_notes(index: KnowledgeIndex, query: str, k: int = 3,
                                retrieval: str = "hybrid", min_content_terms: int = 2) -> dict:
    """Retrieve human-paraphrased notes about identified REAL patents, not full claims.

    This is local search, NOT a live patent-database API or independent verification.
    """
    if not 1 <= len(query.strip()) <= 400:
        raise ValueError("query must be 1-400 characters")
    if not 1 <= k <= 10:
        raise ValueError("k must be from 1 to 10")
    if not 0 <= min_content_terms <= 10:
        raise ValueError("min_content_terms must be 0-10")
    if retrieval == "bm25":
        hits = _cached_bm25(index).search(query, k=len(index.chunks))
    elif retrieval in {"word", "hybrid"}:
        hits = index.search(query, k=len(index.chunks), mode=retrieval)
    else:
        raise ValueError("retrieval must be bm25, word or hybrid")
    # Heuristic evidence gate for out-of-corpus prompts. Rank alone is insufficient:
    # a retriever can select a *best* document even when every match is irrelevant.
    # Rule is deliberately simple and needs calibration on a HELD-OUT real corpus.
    query_terms = set(tokenize(query)) - _QUERY_BOILERPLATE
    filtered: list[dict] = []
    for hit in hits:
        if not hit.doc_id.startswith("patent_real_note_"):
            continue
        content_terms = set(tokenize(index.docs[hit.doc_id][0].text)) - _QUERY_BOILERPLATE
        overlap = query_terms & content_terms
        if len(overlap) < min_content_terms:
            continue
        entry = hit.to_dict()
        entry["content_term_overlap"] = len(overlap)
        filtered.append(entry)
        if len(filtered) >= k:
            break
    return {"source_type": "REAL_PATENT_CURATED_PARAPHRASE_NOT_FULL_TEXT", "retrieval": retrieval,
            "evidence_gate": f"at least {min_content_terms} non-boilerplate query-token overlaps; exploratory, not independently calibrated",
            "count": len(filtered), "results": filtered}


def search_patentsview(query: str, k: int = 3) -> dict:
    """Optional US PatentsView connector; requires a pre-existing official API key.

    Not exercised against the live network by the included tests. The provider
    currently restricts issuance of new keys. No unsupported web scraping.
    """
    key = os.getenv("PATENTSVIEW_API_KEY", "").strip()
    if not key:
        raise RuntimeError("PATENTSVIEW_API_KEY missing; use local synthetic mode or obtain provider access")
    if not 1 <= len(query.strip()) <= 200 or not 1 <= k <= 20:
        raise ValueError("query must be 1-200 characters and k from 1 to 20")
    url = "https://search.patentsview.org/api/v1/patent/"
    body = {"q": {"_text_any": {"patent_abstract": query}},
            "f": ["patent_id", "patent_title", "patent_date", "patent_abstract"],
            "o": {"size": k}}
    req = Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                  headers={"X-Api-Key": key, "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urlopen(req, timeout=12) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"PatentsView request failed ({type(exc).__name__}); no results inferred") from exc
    if not isinstance(result, dict) or result.get("error") is not False:
        raise RuntimeError("PatentsView returned an unsuccessful or unexpected response")
    records = result.get("patents", [])
    if not isinstance(records, list):
        raise RuntimeError("Invalid patents response")
    return {"source_type": "PATENTSVIEW_API_LIVE_UNVERIFIED", "count": len(records),
            "results": [{"patent_id": p.get("patent_id"), "title": p.get("patent_title"),
                         "date": p.get("patent_date"), "abstract": p.get("patent_abstract")}
                        for p in records[:k]]}
