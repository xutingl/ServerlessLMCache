#!/bin/bash
# Test script for SharedDiskBackend-based disaggregated prefill-decode
# This uses disk-based KV cache transfer instead of NIXL P2P

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
MODEL_PATH="${MODEL_PATH:-/workspace/xutingl/downloaded_models/models--meta-llama--Llama-3.1-8B-Instruct/}"
SHARED_DISK_PATH="/tmp/lmcache_shared_kv"

# Port configuration
PREFILL_PORT=7100
DECODE_PORT=7200
PROXY_PORT=9100
ZMQ_PROXY_PORT=7500
NOTIFY_PORT=7300

# Cleanup function
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "disagg_proxy" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    sleep 2
    
    # Clean up shared disk directory
    echo "Cleaning up shared disk directory..."g
    rm -rf "${SHARED_DISK_PATH}"
}

# Set up trap for cleanup
trap cleanup EXIT

# Create shared disk directory
mkdir -p "${SHARED_DISK_PATH}"
echo "Created shared disk directory: ${SHARED_DISK_PATH}"

# Initial cleanup
cleanup

echo "=================================================="
echo "Testing SharedDiskBackend Disaggregated Prefill-Decode"
echo "Model: ${MODEL_PATH}"
echo "Shared Disk Path: ${SHARED_DISK_PATH}"
echo "=================================================="

# Start the proxy server
echo ""
echo "Starting proxy server on port ${PROXY_PORT}..."
python "${SCRIPT_DIR}/disagg_proxy_server.py" \
    --prefill_port ${PREFILL_PORT} \
    --decode_port ${DECODE_PORT} \
    --port ${PROXY_PORT} \
    --zmq_port ${ZMQ_PROXY_PORT} &
PROXY_PID=$!
echo "Proxy PID: ${PROXY_PID}"
sleep 3

# Start the prefiller
echo ""
echo "Starting prefiller on GPU 0, port ${PREFILL_PORT}..."
CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE="${CONFIG_DIR}/lmcache-prefiller-shared-disk.yaml" \
vllm serve "${MODEL_PATH}" \
    --port ${PREFILL_PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector": "LMCacheConnector", "kv_role": "kv_producer"}' &
PREFILL_PID=$!
echo "Prefiller PID: ${PREFILL_PID}"

# Start the decoder
echo ""
echo "Starting decoder on GPU 1, port ${DECODE_PORT}..."
CUDA_VISIBLE_DEVICES=1 \
LMCACHE_CONFIG_FILE="${CONFIG_DIR}/lmcache-decoder-shared-disk.yaml" \
vllm serve "${MODEL_PATH}" \
    --port ${DECODE_PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector": "LMCacheConnector", "kv_role": "kv_consumer"}' &
DECODE_PID=$!
echo "Decoder PID: ${DECODE_PID}"

# Wait for servers to start
echo ""
echo "Waiting for servers to start..."
sleep 60

# Check if processes are running
check_process() {
    if ! kill -0 $1 2>/dev/null; then
        echo "Process $1 ($2) died unexpectedly!"
        return 1
    fi
    return 0
}

check_process ${PROXY_PID} "Proxy" || exit 1
check_process ${PREFILL_PID} "Prefiller" || exit 1
check_process ${DECODE_PID} "Decoder" || exit 1

echo "All processes running!"

# Test completion
test_completion() {
    echo ""
    echo "Testing completion through proxy..."
    
    local response=$(curl -s -X POST "http://localhost:${PROXY_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL_PATH}"'",
            "prompt": "The capital of France is",
            "max_tokens": 50,
            "temperature": 0.7
        }')
    
    echo "Response: ${response}"
    
    # Check if response contains expected fields
    if echo "${response}" | grep -q "choices"; then
        echo "SUCCESS: Got valid completion response"
        return 0
    else
        echo "FAILED: Invalid response"
        return 1
    fi
}

# Run test
test_completion

# Check shared disk directory for cache files
echo ""
echo "Checking shared disk directory for cache files..."
ls -la "${SHARED_DISK_PATH}" 2>/dev/null || echo "No cache files found"

# Show process status
echo ""
echo "Process status:"
ps aux | grep -E "(vllm|disagg_proxy)" | grep -v grep || true

echo ""
echo "=================================================="
echo "Test completed!"
echo "=================================================="

# Keep running for interactive testing
echo ""
echo "Press Ctrl+C to stop all servers..."
wait
