import argparse
import yaml
import logging
import asyncio
from code_flow_graph.mcp_server.server import server

def main():
    parser = argparse.ArgumentParser(
        description="CodeFlowGraph MCP Server - Semantic code analysis and search",
        epilog="Example: python -m code_flow_graph.mcp_server --config custom.yaml"
    )
    parser.add_argument(
        "--config",
        default="code_flow_graph/mcp_server/config/default.yaml",
        help="Path to configuration YAML file (default: code_flow_graph/mcp_server/config/default.yaml)"
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logging.basicConfig(level=logging.INFO)

    # Pass config to server for analyzer initialization
    server.config = config

    logging.info("Server running on stdio")
    try:
        asyncio.run(server.run_stdio_async())
    except KeyboardInterrupt:
        logging.info("Server stopped")

if __name__ == "__main__":
    main()