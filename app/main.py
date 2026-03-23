"""
自定义 Faster-Whisper 服务
- 支持 GPU/CPU 动态切换
- 支持按需自动卸载模型（可配置空闲超时）
- API 兼容 OpenAI /v1/audio/transcriptions 规范
- 支持从本地 whisper-models 目录加载模型
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

# ========== 环境变量配置 ==========
# GPU/CPU 设备选择 (默认: cuda, 可选: cpu)
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
# 计算类型 (GPU 默认: float16, CPU 默认: int8)
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
# CPU 线程数 (仅 CPU 模式有效)
THREADS = int(os.getenv("WHISPER_THREADS", "8"))
# 模型缓存目录
CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", "/root/.cache/huggingface/hub")
# 空闲超时时间（秒），0 表示永不卸载 (默认: 600秒 = 10分钟)
IDLE_TIMEOUT = int(os.getenv("WHISPER_IDLE_TIMEOUT", "600"))
# 默认模型
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")
# HuggingFace 离线模式:
# - "0" = 完全在线（每次检查更新，不推荐）
# - "1" = 完全离线（本地没有就报错）
# - "auto" = 智能模式（本地有就用本地，没有就下载，不检查更新）
HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "auto")

# 设置环境变量（仅在完全离线模式）
if HF_OFFLINE == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"

logger.info(f"配置: DEVICE={DEVICE}, COMPUTE_TYPE={COMPUTE_TYPE}, THREADS={THREADS}")
logger.info(f"缓存目录: {CACHE_DIR}")
logger.info(f"空闲超时: {IDLE_TIMEOUT}秒 ({'禁用自动卸载' if IDLE_TIMEOUT == 0 else '启用自动卸载'})")
offline_mode = "完全离线" if HF_OFFLINE == "1" else ("完全在线" if HF_OFFLINE == "0" else "智能模式(本地优先)")
logger.info(f"HuggingFace 模式: {offline_mode}")

# ========== 全局状态管理 ==========
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
        """加载指定模型"""
        async with self._lock:
            # 如果已有模型且不是请求的模型，先卸载
            if self.model is not None and self.model_name != model_name:
                logger.info(f"切换模型: 卸载 {self.model_name} -> 加载 {model_name}")
                self._unload()

            # 如果未加载，则加载请求的模型
            if self.model is None:
                logger.info(f"正在加载模型: {model_name} 到 {DEVICE}...")
                try:
                    # 智能模式：检查本地是否已有模型
                    local_files_only = False
                    if HF_OFFLINE == "auto":
                        # 检查本地缓存是否存在
                        from huggingface_hub import scan_cache_dir
                        cache = scan_cache_dir(CACHE_DIR)
                        # 检查是否有匹配的模型缓存
                        model_repo = f"Systran/faster-whisper-{model_name}"
                        cached_repos = [repo.repo_id for repo in cache.repos]
                        if model_repo in cached_repos:
                            local_files_only = True
                            logger.info(f"本地已存在模型 {model_name}，使用离线模式")
                        else:
                            logger.info(f"本地无模型 {model_name}，将下载")
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
                    logger.info(f"模型 {model_name} 加载完成")
                except Exception as e:
                    logger.error(f"加载模型失败: {e}")
                    raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

        return self.model

    def _unload(self):
        """内部卸载方法（不加锁）"""
        if self.model is not None:
            logger.info(f"释放模型 {self.model_name} 的系统资源...")
            del self.model
            self.model = None
            self.model_name = None
            gc.collect()

            # GPU 内存清理
            if DEVICE == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.info("GPU 缓存已清理")
                except ImportError:
                    pass

    async def unload(self):
        """公开卸载方法（加锁）"""
        async with self._lock:
            self._unload()

    def should_unload(self):
        """检查是否应该卸载"""
        if IDLE_TIMEOUT == 0:
            return False
        if self.model is None:
            return False
        idle_time = time.time() - self.last_used_time
        return idle_time > IDLE_TIMEOUT


model_state = ModelState()


# ========== 看门狗任务 ==========
async def watchdog_task():
    """定期检查并卸载空闲模型"""
    while True:
        await asyncio.sleep(60)  # 每分钟检查一次
        if model_state.should_unload():
            idle_time = int(time.time() - model_state.last_used_time)
            logger.info(f"模型空闲 {idle_time} 秒，超过阈值 {IDLE_TIMEOUT} 秒，执行卸载")
            await model_state.unload()


# ========== FastAPI 应用 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动看门狗任务
    watchdog = asyncio.create_task(watchdog_task())
    logger.info("自定义 Faster-Whisper 服务已启动")
    yield
    # 关闭时清理
    watchdog.cancel()
    await model_state.unload()
    logger.info("服务已关闭")


app = FastAPI(
    title="Custom Faster-Whisper Server",
    description="兼容 OpenAI API 的 Whisper 语音识别服务，支持 GPU 加速和自动卸载",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """健康检查接口"""
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
    """列出可用模型（模拟 OpenAI API）"""
    # 这里可以扫描本地目录获取实际可用模型
    # 目前返回常见模型列表
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
    兼容 OpenAI API 的语音识别接口

    Args:
        file: 音频文件
        model: 模型名称 (tiny, base, small, medium, large-v2, large-v3 等)
        prompt: 初始提示词
        response_format: 返回格式 (json, verbose_json, text, srt, vtt)
        language: 语言代码 (如 'zh', 'en', 'ja' 等)，None 表示自动检测
        temperature: 采样温度
        timestamp_granularities: 时间戳粒度 (word)
    """
    # 更新最后使用时间
    model_state.update_last_used()

    # 加载请求的模型
    whisper_model = await model_state.load(model)

    # 保存临时文件
    temp_file = f"/tmp/temp_whisper_{time.time()}_{file.filename}"
    try:
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # 执行识别
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

        # 根据请求格式返回
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
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)


def _generate_srt(segments):
    """生成 SRT 字幕格式"""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        srt_content += f"{i}\n"
        srt_content += f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}\n"
        srt_content += f"{segment.text.strip()}\n\n"
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=srt_content)


def _generate_vtt(segments):
    """生成 VTT 字幕格式"""
    vtt_content = "WEBVTT\n\n"
    for segment in segments:
        vtt_content += f"{_format_timestamp(segment.start, vtt=True)} --> {_format_timestamp(segment.end, vtt=True)}\n"
        vtt_content += f"{segment.text.strip()}\n\n"
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=vtt_content)


def _format_timestamp(seconds: float, vtt: bool = False):
    """格式化时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    if vtt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
