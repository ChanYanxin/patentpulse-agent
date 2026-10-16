"""MCP v2 tool server. Requires: pip install -e '.[mcp]' (not needed for API/CLI)."""
from .app import get_agent
from .patents import parse_claims, compare_claims, search_local_patents, search_curated_patent_notes, search_patentsview


def main():
    try:
        from mcp.server import MCPServer  # official MCP Python SDK v2
    except ImportError as exc:
        raise SystemExit("Install the MCP SDK: pip install -e '.[mcp]'") from exc
    mcp = MCPServer("TechIntel Research Tools")

    @mcp.tool()
    def search_research(query: str, k: int = 3) -> list[dict]:
        """Retrieve cited snippets from local technical research documents."""
        return [h.to_dict() for h in get_agent().index.search(query, min(max(k, 1), 5))]

    @mcp.tool()
    def read_research_document(doc_id: str) -> dict:
        """Read research document by identifier from search results."""
        return get_agent().index.get_document(doc_id)

    @mcp.tool()
    def remember_research_note(session_id: str, note: str) -> dict:
        """Save a session-scoped research note."""
        return get_agent().memory.add(session_id, note)

    @mcp.tool()
    def search_patents(query: str, k: int = 3, source: str = "local", retrieval: str = "hybrid") -> dict:
        """Search synthetic local patents, curated real-patent notes, or PatentsView with credentials."""
        if source == "local":
            return search_local_patents(get_agent().index, query, k, retrieval)
        if source == "curated":
            return search_curated_patent_notes(get_agent().index, query, k, retrieval)
        if source == "patentsview":
            return search_patentsview(query, k)
        raise ValueError("source must be local, curated or patentsview")

    @mcp.tool()
    def parse_patent_claims(claims_text: str) -> dict:
        """Parse dependencies in line-numbered English patent claims."""
        return parse_claims(claims_text)

    @mcp.tool()
    def compare_patent_claims(target_claims: str, reference_claims: str) -> dict:
        """Return a lexical claim chart; NOT a legal novelty or infringement determination."""
        return compare_claims(target_claims, reference_claims)

    mcp.run()  # stdio


if __name__ == "__main__":
    main()
