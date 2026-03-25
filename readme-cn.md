# Custom Faster-Whisper Server

中文版 | [English](README.md)

兼容 OpenAI API 的 Whisper 语音识别服务，支持 GPU 加速、自动卸载、stable-ts 高级对齐和 API Key 认证。

## 功能特性

- **GPU/CPU 动态切换** - 支持 CUDA 和 CPU 模式
- **智能模型管理** - 本地有模型就用本地，没有就自动下载（不检查更新）
- **按需自动卸载** - 空闲超时后自动释放内存，可配置或禁用
- **API 兼容** - 完全兼容 OpenAI `/v1/audio/transcriptions` 接口
- **动态模型选择** - 在 API 请求中动态指定模型
- **stable-ts 集成** - 支持高级时间戳对齐和强制对齐功能
- **API Key 认证** - 可选的 API Key 认证，用于公网部署

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
| `API_KEY_REQUIRED` | `false` | 是否启用 API Key 认证 |
| `API_KEYS` | - | 有效的 API Key（分号分隔多个） |
| `ALIGNMENT_AUTO_DETECT_LANGUAGE` | `false` | 强制对齐时自动检测语言（默认：要求用户指定） |

### HF_HUB_OFFLINE 模式

- `auto` - **智能模式（推荐）**：本地有就用本地，没有就下载
- `1` - 完全离线：本地没有就报错
- `0` - 完全在线：每次检查更新

### API Key 认证

启用 API Key 认证用于公网部署：

```bash
# .env 文件
API_KEY_REQUIRED=true
API_KEYS=sk-abc123;sk-xyz789;sk-test123
```

使用 API Key：

```bash
# 方式 1：Authorization 头
curl -H "Authorization: Bearer sk-abc123" \
  http://your-domain.com/v1/audio/transcriptions \
  -F "file=@audio.wav"

# 方式 2：查询参数
curl "http://your-domain.com/v1/audio/transcriptions?api_key=sk-abc123" \
  -F "file=@audio.wav"
```

**公开端点（无需认证）：**
- `/health` - 健康检查
- `/v1/models` - 列出可用模型
- `/docs` - API 文档

## API 使用

### 标准转录（原生引擎）

```bash
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=small" \
  -F "response_format=verbose_json"
```

### 增强转录（stable-ts 引擎）

使用 `stable-ts-` 前缀获得更精准的时间戳对齐：

```bash
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=stable-ts-small" \
  -F "response_format=verbose_json"
```

### 强制对齐

将已有逐字稿与音频时间戳对齐（需要 `stable-ts-*` 模型）：

**注意：** 默认情况下，强制对齐需要指定 `language` 参数。你可以通过设置环境变量 `ALIGNMENT_AUTO_DETECT_LANGUAGE=true` 启用自动语言检测。

```bash
# 方式 1：手动指定语言（默认模式）
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=stable-ts-small" \
  -F "exact_text=这是完整的逐字稿内容，用于时间戳对齐" \
  -F "language=zh" \
  -F "response_format=verbose_json"

# 方式 2：启用自动检测（需设置 ALIGNMENT_AUTO_DETECT_LANGUAGE=true）
# 在 docker-compose.yaml 中添加：
# environment:
#   - ALIGNMENT_AUTO_DETECT_LANGUAGE=true
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=stable-ts-small" \
  -F "exact_text=这是完整的逐字稿内容，用于时间戳对齐" \
  -F "response_format=verbose_json"
```

### 支持的模型

**原生引擎：**
`tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v2`, `large-v3`

**stable-ts 引擎（使用 `stable-ts-` 前缀）：**
`stable-ts-tiny`, `stable-ts-base`, `stable-ts-small`, `stable-ts-medium`, `stable-ts-large-v3`

### 请求参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `file` | File | 音频文件（必填） |
| `model` | String | 模型名称（必填），使用 `stable-ts-*` 启用 stable-ts 引擎 |
| `language` | String | 语言代码（如 `zh`、`en`），不填则自动检测 |
| `prompt` | String | 初始提示词，提供上下文 |
| `response_format` | String | `json`、`verbose_json`、`text`、`srt`、`vtt` |
| `timestamp_granularities[]` | Array | 传 `word` 获取词级时间戳 |
| `exact_text` | String | 完整逐字稿，用于强制对齐（需要 `stable-ts-*` 模型） |

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
  "engine_type": "faster-whisper",
  "idle_timeout": 600,
  "hf_offline": "auto",
  "cache_dir": "/root/.cache/huggingface/hub",
  "api_key_required": false,
  "api_keys_configured": 0
}
```

## 引擎对比

| 特性 | 原生引擎 | stable-ts 引擎 |
|------|----------|----------------|
| 模型名称 | `small`、`large-v3` | `stable-ts-small`、`stable-ts-large-v3` |
| 速度 | 更快 | 略慢 |
| 时间戳质量 | 良好 | 更优（高级对齐算法） |
| 强制对齐 | 不支持 | 支持（使用 `exact_text` 参数） |

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

## 更新日志

### 2026-03-24
- **新增**：使用 `filetype` 库进行音频格式验证
- **新增**：支持 AAC 格式
- **改进**：错误处理 - 对于无效/损坏的音频文件返回 400 而非 500
- **新增**：强制对齐时自动语言检测功能（通过 `ALIGNMENT_AUTO_DETECT_LANGUAGE` 环境变量启用）
- **修复**：强制对齐功能需要指定语言参数的问题
- **支持的格式**：aac, flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm
