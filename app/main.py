"""
Custom Faster-Whisper Server / 自定义 Faster-Whisper 服务
- GPU/CPU dynamic switching / 支持 GPU/CPU 动态切换
- Auto-unload models with configurable idle timeout / 支持按需自动卸载模型（可配置空闲超时）
- OpenAI /v1/audio/transcriptions compatible / API 兼容 OpenAI /v1/audio/transcriptions 规范
- Load models from local whisper-models directory / 支持从本地 whisper-models 目录加载模型
"""

import asyncio
import time
import gc
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ========== Environment Variables / 环境变量配置 ==========
# GPU/CPU device selection (default: cuda, optional: cpu) / GPU/CPU 设备选择 (默认: cuda, 可选: cpu)
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
# Compute type (GPU default: float16, CPU default: int8) / 计算类型 (GPU 默认: float16, CPU 默认: int8)
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
# CPU threads (CPU mode only) / CPU 线程数 (仅 CPU 模式有效)
THREADS = int(os.getenv("WHISPER_THREADS", "8"))
# Model cache directory / 模型缓存目录
CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", "/root/.cache/huggingface/hub")
# Idle timeout in seconds, 0 means never unload (default: 600s = 10min) / 空闲超时时间（秒），0 表示永不卸载 (默认: 600秒 = 10分钟)
IDLE_TIMEOUT = int(os.getenv("WHISPER_IDLE_TIMEOUT", "600"))
# Default model / 默认模型
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")
# HuggingFace offline mode / HuggingFace 离线模式:
# - "0" = Fully online (check updates every time, not recommended) / 完全在线（每次检查更新，不推荐）
# - "1" = Fully offline (error if not found locally) / 完全离线（本地没有就报错）
# - "auto" = Smart mode (use local if exists, download if not, no update checks) / 智能模式（本地有就用本地，没有就下载，不检查更新）
HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "auto")

# Set environment variable (fully offline mode only) / 设置环境变量（仅在完全离线模式）
if HF_OFFLINE == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"

logger.info(f"Config / 配置: DEVICE={DEVICE}, COMPUTE_TYPE={COMPUTE_TYPE}, THREADS={THREADS}")
logger.info(f"Cache dir / 缓存目录: {CACHE_DIR}")
logger.info(f"Idle timeout / 空闲超时: {IDLE_TIMEOUT}s ({'disabled / 禁用自动卸载' if IDLE_TIMEOUT == 0 else 'enabled / 启用自动卸载'})")
offline_mode = "fully offline / 完全离线" if HF_OFFLINE == "1" else ("fully online / 完全在线" if HF_OFFLINE == "0" else "smart mode (local first) / 智能模式(本地优先)")
logger.info(f"HuggingFace mode / HuggingFace 模式: {offline_mode}")

# ========== Global State Management / 全局状态管理 ==========
class ModelState:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.last_used_time = 0
        self._lock = asyncio.Lock()

    def is_loaded(self):
        return self.model is not None

    def update_last_used(self):
        self.last_used_time = time.time()

    async def load(self, model_name: str):
        """Load specified model / 加载指定模型"""
        async with self._lock:
            # If model exists and is different, unload first / 如果已有模型且不是请求的模型，先卸载
            if self.model is not None and self.model_name != model_name:
                logger.info(f"Switching model / 切换模型: unload {self.model_name} -> load {model_name}")
                self._unload()

            # If not loaded, load the requested model / 如果未加载，则加载请求的模型
            if self.model is None:
                logger.info(f"Loading model / 正在加载模型: {model_name} to / 到 {DEVICE}...")
                try:
                    # Smart mode: check if model exists locally / 智能模式：检查本地是否已有模型
                    local_files_only = False
                    if HF_OFFLINE == "auto":
                        # Check local cache / 检查本地缓存是否存在
                        from huggingface_hub import scan_cache_dir
                        cache = scan_cache_dir(CACHE_DIR)
                        # Check if matching model cache exists / 检查是否有匹配的模型缓存
                        model_repo = f"Systran/faster-whisper-{model_name}"
                        cached_repos = [repo.repo_id for repo in cache.repos]
                        if model_repo in cached_repos:
                            local_files_only = True
                            logger.info(f"Model {model_name} exists locally, using offline mode / 本地已存在模型 {model_name}，使用离线模式")
                        else:
                            logger.info(f"Model {model_name} not found locally, will download / 本地无模型 {model_name}，将下载")
                    elif HF_OFFLINE == "1":
                        local_files_only = True

                    self.model = WhisperModel(
                        model_size_or_path=model_name,
                        device=DEVICE,
                        device_index=0 if DEVICE == "cuda" else None,
                        compute_type=COMPUTE_TYPE,
                        cpu_threads=THREADS if DEVICE == "cpu" else 0,
                        num_workers=1,
                        download_root=CACHE_DIR,
                        local_files_only=local_files_only,
                    )
                    self.model_name = model_name
                    self.last_used_time = time.time()
                    logger.info(f"Model {model_name} loaded successfully / 模型 {model_name} 加载完成")
                except Exception as e:
                    logger.error(f"Model load failed / 加载模型失败: {e}")
                    raise HTTPException(status_code=500, detail=f"Model load failed / 模型加载失败: {str(e)}")

        return self.model

    def _unload(self):
        """Internal unload method (no lock) / 内部卸载方法（不加锁）"""
        if self.model is not None:
            logger.info(f"Releasing resources for model {self.model_name}... / 释放模型 {self.model_name} 的系统资源...")
            del self.model
            self.model = None
            self.model_name = None
            gc.collect()

            # GPU memory cleanup / GPU 内存清理
            if DEVICE == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared / GPU 缓存已清理")
                except ImportError:
                    pass

    async def unload(self):
        """Public unload method (with lock) / 公开卸载方法（加锁）"""
        async with self._lock:
            self._unload()

    def should_unload(self):
        """Check if model should be unloaded / 检查是否应该卸载"""
        if IDLE_TIMEOUT == 0:
            return False
        if self.model is None:
            return False
        idle_time = time.time() - self.last_used_time
        return idle_time > IDLE_TIMEOUT


model_state = ModelState()


# ========== Watchdog Task / 看门狗任务 ==========
async def watchdog_task():
    """Periodically check and unload idle models / 定期检查并卸载空闲模型"""
    while True:
        await asyncio.sleep(60)  # Check every minute / 每分钟检查一次
        if model_state.should_unload():
            idle_time = int(time.time() - model_state.last_used_time)
            logger.info(f"Model idle for {idle_time}s, exceeding threshold {IDLE_TIMEOUT}s, unloading / 模型空闲 {idle_time} 秒，超过阈值 {IDLE_TIMEOUT} 秒，执行卸载")
            await model_state.unload()


# ========== FastAPI Application / FastAPI 应用 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management / 应用生命周期管理"""
    # Start watchdog task / 启动看门狗任务
    watchdog = asyncio.create_task(watchdog_task())
    logger.info("Custom Faster-Whisper server started / 自定义 Faster-Whisper 服务已启动")
    yield
    # Cleanup on shutdown / 关闭时清理
    watchdog.cancel()
    await model_state.unload()
    logger.info("Server shutdown / 服务已关闭")


app = FastAPI(
    title="Custom Faster-Whisper Server",
    description="OpenAI API compatible Whisper speech recognition service with GPU acceleration and auto-unload / 兼容 OpenAI API 的 Whisper 语音识别服务，支持 GPU 加速和自动卸载",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint / 健康检查接口"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "threads": THREADS,
        "model_loaded": model_state.is_loaded(),
        "model_name": model_state.model_name,
        "idle_timeout": IDLE_TIMEOUT,
        "hf_offline": HF_OFFLINE,
        "cache_dir": CACHE_DIR
    }


@app.get("/v1/models")
async def list_models():
    """List available models (simulates OpenAI API) / 列出可用模型（模拟 OpenAI API）"""
    # Can scan local directory for actual available models / 这里可以扫描本地目录获取实际可用模型
    # Currently returns common model list / 目前返回常见模型列表
    return {
        "object": "list",
        "data": [
            {"id": "tiny", "object": "model", "owned_by": "Systran"},
            {"id": "tiny.en", "object": "model", "owned_by": "Systran"},
            {"id": "base", "object": "model", "owned_by": "Systran"},
            {"id": "base.en", "object": "model", "owned_by": "Systran"},
            {"id": "small", "object": "model", "owned_by": "Systran"},
            {"id": "small.en", "object": "model", "owned_by": "Systran"},
            {"id": "medium", "object": "model", "owned_by": "Systran"},
            {"id": "medium.en", "object": "model", "owned_by": "Systran"},
            {"id": "large-v1", "object": "model", "owned_by": "Systran"},
            {"id": "large-v2", "object": "model", "owned_by": "Systran"},
            {"id": "large-v3", "object": "model", "owned_by": "Systran"},
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    language: str = Form(None),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] = Form(None),
):
    """
    OpenAI API compatible speech recognition endpoint / 兼容 OpenAI API 的语音识别接口

    Args:
        file: Audio file / 音频文件
        model: Model name (tiny, base, small, medium, large-v2, large-v3, etc.) / 模型名称
        prompt: Initial prompt / 初始提示词
        response_format: Response format (json, verbose_json, text, srt, vtt) / 返回格式
        language: Language code (e.g., 'zh', 'en', 'ja'), None for auto-detect / 语言代码，None 表示自动检测
        temperature: Sampling temperature / 采样温度
        timestamp_granularities: Timestamp granularity (word) / 时间戳粒度
    """
    # Update last used time / 更新最后使用时间
    model_state.update_last_used()

    # Load requested model / 加载请求的模型
    whisper_model = await model_state.load(model)

    # Save temp file / 保存临时文件
    temp_file = f"/tmp/temp_whisper_{time.time()}_{file.filename}"
    try:
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Execute transcription / 执行识别
        segments, info = whisper_model.transcribe(
            temp_file,
            initial_prompt=prompt,
            language=language,
            word_timestamps=True,
            beam_size=1,
            temperature=temperature,
        )

        # 组装结果
        full_text = ""
        words_list = []

        for segment in segments:
            full_text += segment.text
            if segment.words:
                for word in segment.words:
                    words_list.append({
                        "word": word.word.strip(),
                        "start": round(word.start, 3),
                        "end": round(word.end, 3)
                    })

        # Return based on requested format / 根据请求格式返回
        if response_format == "verbose_json":
            return JSONResponse(content={
                "task": "transcribe",
                "language": info.language,
                "duration": round(info.duration, 2),
                "text": full_text.strip(),
                "words": words_list
            })
        elif response_format == "json":
            return JSONResponse(content={"text": full_text.strip()})
        elif response_format == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=full_text.strip())
        elif response_format == "srt":
            return _generate_srt(segments)
        elif response_format == "vtt":
            return _generate_vtt(segments)
        else:
            return JSONResponse(content={"text": full_text.strip()})

    finally:
        # Cleanup temp file / 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)


def _generate_srt(segments):
    """Generate SRT subtitle format / 生成 SRT 字幕格式"""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        srt_content += f"{i}\n"
        srt_content += f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}\n"
        srt_content += f"{segment.text.strip()}\n\n"
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=srt_content)


def _generate_vtt(segments):
    """Generate VTT subtitle format / 生成 VTT 字幕格式"""
    vtt_content = "WEBVTT\n\n"
    for segment in segments:
        vtt_content += f"{_format_timestamp(segment.start, vtt=True)} --> {_format_timestamp(segment.end, vtt=True)}\n"
        vtt_content += f"{segment.text.strip()}\n\n"
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=vtt_content)


def _format_timestamp(seconds: float, vtt: bool = False):
    """Format timestamp / 格式化时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    if vtt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
