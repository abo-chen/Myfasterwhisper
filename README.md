# Custom Faster-Whisper Server

兼容 OpenAI API 的 Whisper 语音识别服务，支持 GPU 加速和自动卸载。

## 功能特性

- **GPU/CPU 动态切换** - 支持 CUDA 和 CPU 模式
- **智能模型管理** - 本地有模型就用本地，没有就自动下载（不检查更新）
- **按需自动卸载** - 空闲超时后自动释放内存，可配置或禁用
- **API 兼容** - 完全兼容 OpenAI `/v1/audio/transcriptions` 接口
- **动态模型选择** - 在 API 请求中动态指定模型

## 快速开始

### Docker 部署（推荐）

```bash
# 克隆项目
git clone <repo-url>
cd fasterwhisper

# 配置环境变量
cp .env.example .env
vim .env

# 启动服务
docker compose up -d --build
```

### 本地运行

```bash
# 安装 uv
pip install uv

# 运行
./run.sh
```

## 配置说明

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `WHISPER_DEVICE` | `cuda` | 设备选择：`cuda` 或 `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` | GPU 用 `float16`，CPU 用 `int8` |
| `WHISPER_THREADS` | `8` | CPU 线程数（仅 CPU 模式） |
| `WHISPER_MODEL` | `small` | 默认模型名称 |
| `WHISPER_IDLE_TIMEOUT` | `600` | 空闲超时（秒），0=永不卸载 |
| `HF_HUB_OFFLINE` | `auto` | 离线模式（见下方） |

### HF_HUB_OFFLINE 模式

- `auto` - **智能模式（推荐）**：本地有就用本地，没有就下载
- `1` - 完全离线：本地没有就报错
- `0` - 完全在线：每次检查更新

## API 使用

### 语音识别

```bash
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=small" \
  -F "response_format=json"
```

### 支持的模型

`tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v2`, `large-v3`

### 返回格式

- `json` - 简单文本
- `verbose_json` - 包含时间戳和词级信息
- `text` - 纯文本
- `srt` - SRT 字幕
- `vtt` - VTT 字幕

## 健康检查

```bash
curl http://localhost:5012/health
```

返回：
```json
{
  "status": "healthy",
  "device": "cuda",
  "compute_type": "float16",
  "model_loaded": true,
  "model_name": "small",
  "idle_timeout": 600,
  "hf_offline": "auto"
}
```

## 项目结构

```
fasterwhisper/
├── app/
│   └── main.py          # 核心服务代码
├── whisper-models/      # 模型缓存目录
├── Dockerfile           # Docker 镜像构建
├── docker-compose.yaml  # Docker Compose 配置
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量示例
└── run.sh              # 本地运行脚本
```
