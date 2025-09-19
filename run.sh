#!/usr/bin/env bash

PYTHONPATH=$(dirname $0)/code_flow_graph:$PYTHONPATH 

(cd $(dirname $0) && uv run code_flow_graph_mcp_server $@)