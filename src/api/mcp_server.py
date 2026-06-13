# Exposes the RAG pipeline as an MCP server — allows AI agents to discover and call it as a tool.

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from src.llm.generator import AnswerGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.core.logger import get_logger

logger = get_logger(__name__)

# MCP server instance — name identifies this server to connecting agents
mcp = Server("wealth-ai-assistant")

# Initialize pipeline once — shared across all tool calls
generator = AnswerGenerator(prompt_version="v1")
ingestion = IngestionPipeline()


# --- Tool Definitions ---

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    """Registers all available tools — agents call this to discover what this server can do."""
    return [
        Tool(
            name="query_documents",
            description=(
                "Answer questions about financial documents including prospectuses, "
                "portfolio reports, and market summaries. Returns an answer with "
                "citations showing the exact source document and page number. "
                "Use this when a wealth advisor asks about fund performance, "
                "fees, risk factors, or investment strategies."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The financial question to answer",
                        "minLength": 5,
                        "maxLength": 1000,
                    },
                    "prompt_version": {
                        "type": "string",
                        "description": "Prompt version to use — defaults to v1",
                        "default": "v1",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="list_documents",
            description=(
                "Lists all financial documents currently available in the knowledge base. "
                "Use this when the advisor wants to know which documents can be queried, "
                "or to verify a specific document has been ingested."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# --- Tool Handlers ---

@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Routes incoming tool calls to the correct handler based on tool name."""
    logger.info(f"MCP tool called: '{name}' with args: {arguments}")

    if name == "query_documents":
        return await _handle_query(arguments)

    elif name == "list_documents":
        return await _handle_list_documents()

    else:
        logger.warning(f"Unknown tool requested: '{name}'")
        return [TextContent(
            type="text",
            text=f"Unknown tool: '{name}'. Available tools: query_documents, list_documents",
        )]


async def _handle_query(arguments: dict) -> list[TextContent]:
    """Runs the full RAG pipeline for a question and formats the result for the agent."""
    question = arguments.get("question", "").strip()

    if not question:
        return [TextContent(type="text", text="Error: 'question' field is required.")]

    try:
        # Run the synchronous generator in a thread so we don't block the async event loop
        response = await asyncio.to_thread(generator.answer, question)

        # Format citations as readable text for the agent
        citation_lines = []
        for i, citation in enumerate(response.citations, start=1):
            citation_lines.append(
                f"[{i}] {citation.document}, Page {citation.page_number}\n"
                f"     \"{citation.chunk_text[:150]}\""
            )

        citations_block = (
            "\n".join(citation_lines)
            if citation_lines
            else "No citations available."
        )

        result = (
            f"Answer:\n{response.answer}\n\n"
            f"Sources:\n{citations_block}\n\n"
            f"Metadata:\n"
            f"  Model: {response.model_used}\n"
            f"  Prompt version: {response.prompt_version}"
        )

        return [TextContent(type="text", text=result)]

    except ValueError as e:
        # Injection attempt or invalid query
        return [TextContent(type="text", text=f"Query rejected: {e}")]

    except Exception as e:
        logger.error(f"Error in query_documents tool: {e}")
        return [TextContent(type="text", text=f"Error processing query: {e}")]


async def _handle_list_documents() -> list[TextContent]:
    """Returns a summary of all ingested documents from ChromaDB."""
    try:
        collection = ingestion.collection
        results = collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return [TextContent(
                type="text",
                text="No documents currently ingested in the knowledge base.",
            )]

        # Count chunks per document
        doc_summary = {}
        for metadata in results["metadatas"]:
            source = metadata.get("source_file", "unknown")
            doc_summary[source] = doc_summary.get(source, 0) + 1

        lines = [f"Documents in knowledge base ({len(doc_summary)} files):\n"]
        for filename, chunk_count in doc_summary.items():
            lines.append(f"  • {filename} — {chunk_count} chunks")

        lines.append(f"\nTotal chunks: {sum(doc_summary.values())}")

        return [TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        logger.error(f"Error in list_documents tool: {e}")
        return [TextContent(type="text", text=f"Error listing documents: {e}")]


# --- Entry Point ---

async def run():
    """Starts the MCP server on stdio — agents connect via standard input/output."""
    logger.info("Starting Wealth AI MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(
            read_stream,
            write_stream,
            mcp.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(run())