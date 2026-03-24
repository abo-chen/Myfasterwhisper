# Custom Faster-Whisper Server

[中文版](readme-cn.md) | English

OpenAI API compatible Whisper speech recognition service with GPU acceleration, auto-unload, stable-ts support, and API key authentication.

## Features

- **GPU/CPU Switching** - Support both CUDA and CPU modes
- **Smart Model Management** - Use local models if available, download if not (no update checks)
- **Auto Unload** - Automatically free memory after idle timeout, configurable or disable
- **API Compatible** - Fully compatible with OpenAI `/v1/audio/transcriptions` endpoint
- **Dynamic Model Selection** - Specify model in API request
- **stable-ts Integration** - Advanced timestamp alignment and forced alignment support
- **API Key Authentication** - Optional API key authentication for secure deployment

## Quick Start

### Docker Deployment (Recommended)

```bash
# Clone project
git clone <repo-url>
cd fasterwhisper

# Configure environment
cp .env.example .env
vim .env

# Start service
docker compose up -d --build
```

### Local Running

```bash
# Install uv
pip install uv

# Run
./run.sh
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `WHISPER_DEVICE` | `cuda` | Device: `cuda` or `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` | GPU: `float16`, CPU: `int8` |
| `WHISPER_THREADS` | `8` | CPU threads (CPU mode only) |
| `WHISPER_MODEL` | `small` | Default model name |
| `WHISPER_IDLE_TIMEOUT` | `600` | Idle timeout (seconds), 0=never unload |
| `HF_HUB_OFFLINE` | `auto` | Offline mode (see below) |
| `API_KEY_REQUIRED` | `false` | Enable API key authentication |
| `API_KEYS` | - | Valid API keys (separated by semicolon) |

### HF_HUB_OFFLINE Modes

- `auto` - **Smart mode (recommended)**: Use local if exists, download if not
- `1` - Fully offline: Error if not found locally
- `0` - Fully online: Check for updates every time

### API Key Authentication

Enable API key authentication for public deployment:

```bash
# .env file
API_KEY_REQUIRED=true
API_KEYS=sk-abc123;sk-xyz789;sk-test123
```

Usage with API key:

```bash
# Method 1: Authorization header
curl -H "Authorization: Bearer sk-abc123" \
  http://your-domain.com/v1/audio/transcriptions \
  -F "file=@audio.wav"

# Method 2: Query parameter
curl "http://your-domain.com/v1/audio/transcriptions?api_key=sk-abc123" \
  -F "file=@audio.wav"
```

**Public endpoints (no authentication required):**
- `/health` - Health check
- `/v1/models` - List available models
- `/docs` - API documentation

## API Usage

### Standard Transcription (Native Engine)

```bash
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=small" \
  -F "response_format=verbose_json"
```

### Enhanced Transcription (stable-ts Engine)

Use `stable-ts-` prefix for improved timestamp alignment:

```bash
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=stable-ts-small" \
  -F "response_format=verbose_json"
```

### Forced Alignment

Align an existing transcript with audio timestamps (requires `stable-ts-*` model):

```bash
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=stable-ts-small" \
  -F "exact_text=This is the complete transcript for alignment" \
  -F "language=en" \
  -F "response_format=verbose_json"
```

### Supported Models

**Native Engine:**
`tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v2`, `large-v3`

**stable-ts Engine (prefix with `stable-ts-`):**
`stable-ts-tiny`, `stable-ts-base`, `stable-ts-small`, `stable-ts-medium`, `stable-ts-large-v3`

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | File | Audio file (required) |
| `model` | String | Model name (required), use `stable-ts-*` for stable-ts engine |
| `language` | String | Language code (e.g., `zh`, `en`), auto-detect if omitted |
| `prompt` | String | Initial prompt for context |
| `response_format` | String | `json`, `verbose_json`, `text`, `srt`, `vtt` |
| `timestamp_granularities[]` | Array | `word` for word-level timestamps |
| `exact_text` | String | Full transcript for forced alignment (requires `stable-ts-*` model) |

### Response Formats

- `json` - Simple text
- `verbose_json` - With timestamps and word-level info
- `text` - Plain text
- `srt` - SRT subtitles
- `vtt` - VTT subtitles

## Health Check

```bash
curl http://localhost:5012/health
```

Response:
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

## Engine Comparison

| Feature | Native Engine | stable-ts Engine |
|---------|---------------|------------------|
| Model Name | `small`, `large-v3` | `stable-ts-small`, `stable-ts-large-v3` |
| Speed | Faster | Slightly slower |
| Timestamp Quality | Good | Better (advanced alignment) |
| Forced Alignment | No | Yes (`exact_text` parameter) |

## Project Structure

```
fasterwhisper/
├── app/
│   └── main.py          # Core service code
├── whisper-models/      # Model cache directory
├── Dockerfile           # Docker image build
├── docker-compose.yaml  # Docker Compose config
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables example
└── run.sh              # Local run script
```
