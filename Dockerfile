#syntax=docker/dockerfile:1.7
# 多阶段构建，支持 CUDA 和 CPU
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# 安装系统依赖和 uv（使用 BuildKit 缓存挂载加速）
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-venv \
    ffmpeg \
    curl \
    ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（官方方式）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 使用 uv 安装依赖（使用 BuildKit 缓存挂载加速）
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# 复制应用代码
COPY app/ ./app/

# 创建缓存目录
RUN mkdir -p /root/.cache/huggingface/hub

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
