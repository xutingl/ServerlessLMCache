#!/bin/bash

# Test script for pipelined layerwise KV cache transfer with disaggregated prefill-decode
# Uses local model and enables layerwise pipelining

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model configuration - use HF cache
export HF_HUB_CACHE="/workspace/xutingl/downloaded_models"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Can also use complete local model:
# MODEL="/workspace/xutingl/downloaded_models/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8/"

# For correct KV cache transfer, ensure all processes use the same PYTHONHASHSEED
export PYTHONHASHSEED=0

# GPU assignments
export PREFILLER_DEVICE_ID="${PREFILLER_DEVICE_ID:-0}"
export DECODER_DEVICE_ID="${DECODER_DEVICE_ID:-1}"

PIDS=()

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM USR1
    
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null
        fi
    done
    
    sleep 2
    
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done
    
    echo "All processes stopped."
    exit 0
}

wait_for_server() {
    local port=$1
    local timeout_seconds=600
    local start_time=$(date +%s)
    
    echo "Waiting for server on port $port..."
    
    while true; do
        if curl -s "localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "Server on port $port is ready!"
            return 0
        fi
        
        local now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then
            echo "Timeout waiting for server on port $port"
            return 1
        fi
        
        sleep 2
    done
}

run_prefiller() {
    local config_file=$SCRIPT_DIR/configs/lmcache-prefiller-layerwise.yaml
    
    echo "Starting prefiller on GPU $PREFILLER_DEVICE_ID with layerwise pipelining..."
    
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=$config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=$PREFILLER_DEVICE_ID \
        HF_HUB_CACHE=$HF_HUB_CACHE \
        vllm serve $MODEL \
        --port 7100 \
        --disable-log-requests \
        --enforce-eager \
        --no-enable-prefix-caching \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}' \
        2>&1 | tee prefiller.log &
    
    local pid=$!
    PIDS+=($pid)
    echo "Prefiller started with PID $pid"
}

run_decoder() {
    local config_file=$SCRIPT_DIR/configs/lmcache-decoder-layerwise.yaml
    
    echo "Starting decoder on GPU $DECODER_DEVICE_ID with layerwise pipelining and save_decode_cache..."
    
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=$config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=$DECODER_DEVICE_ID \
        HF_HUB_CACHE=$HF_HUB_CACHE \
        vllm serve $MODEL \
        --port 7200 \
        --disable-log-requests \
        --enforce-eager \
        --no-enable-prefix-caching \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}' \
        2>&1 | tee decoder.log &
    
    local pid=$!
    PIDS+=($pid)
    echo "Decoder started with PID $pid"
}

run_proxy() {
    echo "Starting proxy server..."
    
    python3 $SCRIPT_DIR/../disagg_proxy_server.py \
        --host localhost \
        --port 9100 \
        --prefiller-host localhost \
        --prefiller-port 7100 \
        --num-prefillers 1 \
        --decoder-host localhost \
        --decoder-port 7200 \
        --decoder-init-port 7300 \
        --decoder-alloc-port 7400 \
        --proxy-host localhost \
        --proxy-port 7500 \
        --num-decoders 1 \
        2>&1 | tee proxy.log &
    
    local pid=$!
    PIDS+=($pid)
    echo "Proxy started with PID $pid"
}

test_completion() {
    echo ""
    echo "============================================"
    echo "Testing completion with disaggregated prefill..."
    echo "============================================"
    
    curl -s http://localhost:9100/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "prompt": "Hello, my name is",
            "max_tokens": 50,
            "temperature": 0.7
        }' | python3 -m json.tool
    
    echo ""
    echo "Test complete!"
}

main() {
    trap cleanup INT TERM USR1
    
    echo "============================================"
    echo "Layerwise Pipelined KV Transfer Test"
    echo "============================================"
    echo "Model: $MODEL"
    echo "Prefiller GPU: $PREFILLER_DEVICE_ID"
    echo "Decoder GPU: $DECODER_DEVICE_ID"
    echo "Layerwise mode: ENABLED"
    echo "Save decode cache: ENABLED"
    echo "============================================"
    echo ""
    
    # Start proxy first
    run_proxy
    sleep 2
    
    # Start decoder 
    run_decoder
    
    # Start prefiller
    run_prefiller
    
    # Wait for servers to be ready
    echo ""
    echo "Waiting for servers to start..."
    if ! wait_for_server 7200; then
        echo "Decoder failed to start. Check decoder.log"
        cleanup
    fi
    
    if ! wait_for_server 7100; then
        echo "Prefiller failed to start. Check prefiller.log"
        cleanup
    fi
    
    if ! wait_for_server 9100; then
        echo "Proxy failed to start. Check proxy.log"
        cleanup
    fi
    
    echo ""
    echo "==================================================="
    echo "All servers are up!"
    echo "Proxy: http://localhost:9100"
    echo "Prefiller: http://localhost:7100"
    echo "Decoder: http://localhost:7200"
    echo "==================================================="
    
    # Run a test completion
    sleep 3
    test_completion
    
    echo ""
    echo "Press Ctrl-C to stop all servers..."
    
    # Keep running
    while true; do
        sleep 1
    done
}

main
