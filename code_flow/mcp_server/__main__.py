import argparse
import logging
import sys
import asyncio
from code_flow.mcp_server.server import server
from code_flow.core.config import load_config

def main():
    parser = argparse.ArgumentParser(
        description="CodeFlowGraph MCP Server - Semantic code analysis and search",
        epilog="Example: python -m code_flow.mcp_server --config custom.yaml"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration YAML file (default: codeflow.config.yaml)"
    )
    # Add other overrides if useful, e.g. --port, --debug
    
    args = parser.parse_args()
    
    # Load configuration
    # We pass args as dict, filtering out None
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config = load_config(config_path=args.config, cli_args=cli_args)
    
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Pass config dict to server for analyzer initialization
    # The analyzer expects a dict, so we dump the pydantic model
    config_dict = config.model_dump()
    config_dict.update({
        "chroma_dir": str(config.chroma_dir()),
        "memory_dir": str(config.memory_dir()),
        "reports_dir": str(config.reports_dir()),
        "cache_dir": str(config.cache_dir()),
    })
    server.config = config_dict

    logging.info(f"Server starting with config: {server.config}")
    logging.info("Server running on stdio")
    try:
        asyncio.run(server.run_stdio_async())
    except KeyboardInterrupt:
        logging.info("Server stopped")

if __name__ == "__main__":
    main()
