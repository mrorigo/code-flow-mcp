import sys
import json

# Send initialize message
initialize = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    }
}
json.dump(initialize, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send initialized notification
initialized = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized",
    "params": {}
}
json.dump(initialized, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send handshake tool call
handshake_call = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "on_handshake",
        "arguments": {"version": "2025.6"}
    }
}
json.dump(handshake_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send listResources tool call
list_resources_call = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "listResources",
        "arguments": {}
    }
}
json.dump(list_resources_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send ping tool call
ping_call = {
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
        "name": "ping",
        "arguments": {"message": "test"}
    }
}
json.dump(ping_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send semantic_search tool call
semantic_search_call = {
    "jsonrpc": "2.0",
    "id": 5,
    "method": "tools/call",
    "params": {
        "name": "semantic_search",
        "arguments": {"query": "test"}
    }
}
json.dump(semantic_search_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send get_call_graph tool call
get_call_graph_call = {
    "jsonrpc": "2.0",
    "id": 6,
    "method": "tools/call",
    "params": {
        "name": "get_call_graph",
        "arguments": {}
    }
}
json.dump(get_call_graph_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send get_function_metadata tool call
get_function_metadata_call = {
    "jsonrpc": "2.0",
    "id": 7,
    "method": "tools/call",
    "params": {
        "name": "get_function_metadata",
        "arguments": {"fqn": "code_flow.cli.code_flow.main"}  # Known FQN from index
    }
}
json.dump(get_function_metadata_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send query_entry_points tool call
query_entry_points_call = {
    "jsonrpc": "2.0",
    "id": 8,
    "method": "tools/call",
    "params": {
        "name": "query_entry_points",
        "arguments": {}
    }
}
json.dump(query_entry_points_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send generate_mermaid_graph tool call
generate_mermaid_graph_call = {
    "jsonrpc": "2.0",
    "id": 9,
    "method": "tools/call",
    "params": {
        "name": "generate_mermaid_graph",
        "arguments": {}
    }
}
json.dump(generate_mermaid_graph_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send update_context tool call
update_context_call = {
    "jsonrpc": "2.0",
    "id": 10,
    "method": "tools/call",
    "params": {
        "name": "update_context",
        "arguments": {"test": "val"}
    }
}
json.dump(update_context_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send get_context tool call
get_context_call = {
    "jsonrpc": "2.0",
    "id": 11,
    "method": "tools/call",
    "params": {
        "name": "get_context",
        "arguments": {}
    }
}
json.dump(get_context_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()

# Send file touch to trigger watcher (if configured)
# This simulates a file change that the watcher would detect
import os
import time
test_file = "test_file.py"
if os.path.exists(test_file):
    # Touch the file to update its modification time
    os.utime(test_file, None)
    print(f"Touched {test_file} to trigger file watcher", file=sys.stderr)

# Wait a moment for watcher to process
time.sleep(0.1)

# Send re-search after file touch
re_search_call = {
    "jsonrpc": "2.0",
    "id": 12,
    "method": "tools/call",
    "params": {
        "name": "semantic_search",
        "arguments": {"query": "test function", "n_results": 3}
    }
}
json.dump(re_search_call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()