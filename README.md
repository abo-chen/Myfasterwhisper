# Custom Faster-Whisper Server

English | [中文版](readme-cn.md)

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

# Start GPU 0 service
docker compose up -d --build

# (Optional) Start GPU 1 service when needed
docker compose --profile gpu1 up -d
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
| `ALIGNMENT_AUTO_DETECT_LANGUAGE` | `false` | Auto-detect language for forced alignment (default: require user to specify) |

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

### VAD Voice Activity Detection (Native Engine)

Enable VAD to filter out non-speech segments before transcription, improving accuracy and speed for audio with long silences (native engine only):

```bash
# Enable VAD with recommended defaults
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=small" \
  -F "vad_filter=true"

# Custom VAD parameters
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=small" \
  -F "vad_filter=true" \
  -F "vad_min_silence_duration_ms=300" \
  -F "vad_speech_pad_ms=200" \
  -F "vad_threshold=0.6"
```

### Forced Alignment

Align an existing transcript with audio timestamps (requires `stable-ts-*` model):

**Note:** By default, the `language` parameter is **required** for forced alignment. You can enable automatic language detection by setting `ALIGNMENT_AUTO_DETECT_LANGUAGE=true` in your environment.

**Text Sanitization:** By default, the `exact_text` is automatically sanitized to prevent alignment issues caused by hidden characters (zero-width characters, non-breaking spaces, etc.). You can disable this by setting `sanitize_text=false`.

```bash
# Method 1: Specify language (default mode)
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=stable-ts-small" \
  -F "exact_text=This is the complete transcript for alignment" \
  -F "language=en" \
  -F "response_format=verbose_json"

# Method 2: Enable auto-detection (set ALIGNMENT_AUTO_DETECT_LANGUAGE=true)
# In docker-compose.yaml:
# environment:
#   - ALIGNMENT_AUTO_DETECT_LANGUAGE=true
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=stable-ts-small" \
  -F "exact_text=This is the complete transcript for alignment" \
  -F "response_format=verbose_json"

# Method 3: Disable text sanitization (for special formatting like poetry)
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=stable-ts-small" \
  -F "exact_text=This is the complete transcript for alignment" \
  -F "language=en" \
  -F "sanitize_text=false" \
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
| `sanitize_text` | String | Enable text sanitization for forced alignment: `true` (default) or `false` |
| `vad_filter` | String | Enable VAD filter (native engine only): `true` or `false` (default) |
| `vad_min_silence_duration_ms` | Int | Min silence duration in ms to split segments (default: `500`) |
| `vad_speech_pad_ms` | Int | Padding in ms around speech segments (default: `400`) |
| `vad_threshold` | Float | VAD confidence threshold (default: `0.5`) |

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
| VAD Filter | Yes | No |

## Multi-GPU Deployment

When you have multiple GPUs, you can start an additional instance on GPU 1 to increase throughput by processing requests in parallel. Both services are defined in a single `docker-compose.yml`, with GPU 1 service using a profile so it only starts on demand.

### Start / Stop

```bash
# Start services
docker compose up -d                                      # GPU 0 only
docker compose --profile gpu1 up -d whisper-gpu1          # GPU 1 only
docker compose --profile gpu1 up -d                       # Both GPU 0 + GPU 1

# Stop services
docker compose stop whisper-gpu0                          # GPU 0 only
docker compose --profile gpu1 stop whisper-gpu1           # GPU 1 only
docker compose down                                       # Stop all
```

### Usage

The main service runs on port 5012 (GPU 0), and the additional instance runs on port 5013 (GPU 1). Distribute requests across both ports:

```bash
# Main service (GPU 0)
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=large-v3" \
  -F "response_format=srt"

# Additional instance (GPU 1)
curl -X POST http://localhost:5013/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=large-v3" \
  -F "response_format=srt"
```

## Project Structure

```
fasterwhisper/
├── app/
│   └── main.py                     # Core service code
├── whisper-models/                 # Model cache directory
├── Dockerfile                      # Docker image build
├── docker-compose.yml              # GPU 0 (default) + GPU 1 (profile: gpu1)
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables example
└── run.sh                          # Local run script
```

## Changelog

### 2026-04-30
- **Added**: VAD (Voice Activity Detection) filter support for native engine
  - Filter out non-speech segments before transcription, improving accuracy and speed
  - Configurable via `vad_filter`, `vad_min_silence_duration_ms`, `vad_speech_pad_ms`, `vad_threshold` parameters
  - Disabled by default, only applies to native faster-whisper engine

### 2026-03-29
- **Added**: Text sanitization for forced alignment to prevent tokenization crashes
  - Removes zero-width characters, non-breaking spaces, and control characters
  - Unicode NFKC normalization (full-width to half-width conversion)
  - Preserves line breaks for subtitle segmentation
  - Optional via `sanitize_text` parameter (default: `true`)
- **Fixed**: Alignment issues caused by hidden characters copied from PDFs/web pages

### 2026-03-24
- **Added**: Audio format validation using `filetype` library
- **Added**: Support for AAC format
- **Improved**: Error handling - return 400 instead of 500 for invalid/corrupted audio files
- **Added**: Auto language detection for forced alignment (via `ALIGNMENT_AUTO_DETECT_LANGUAGE` env var)
- **Fixed**: Language parameter requirement for forced alignment feature
- **Supported formats**: aac, flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm
