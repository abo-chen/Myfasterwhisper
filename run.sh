#!/bin/bash
# 本地运行脚本（使用 uv）

# 加载环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 设置默认值
export WHISPER_DEVICE=${WHISPER_DEVICE:-cuda}
export WHISPER_COMPUTE_TYPE=${WHISPER_COMPUTE_TYPE:-float16}
export WHISPER_THREADS=${WHISPER_THREADS:-8}
export WHISPER_MODEL=${WHISPER_MODEL:-small}
export WHISPER_IDLE_TIMEOUT=${WHISPER_IDLE_TIMEOUT:-600}
export WHISPER_CACHE_DIR=${WHISPER_CACHE_DIR:-./whisper-models}

echo "=== Custom Faster-Whisper Server ==="
echo "设备: $WHISPER_DEVICE"
echo "计算类型: $WHISPER_COMPUTE_TYPE"
echo "空闲超时: $WHISPER_IDLE_TIMEOUT 私"
echo "===================================="

# 使用 uv 运行
uv run --with faster-whisper --with fastapi --with uvicorn --with python-multipart \
    uvicorn app.main:app --host 0.0.0.0 --port 5012 --reload
