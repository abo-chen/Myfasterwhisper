# Custom Faster-Whisper Server

[中文版](readme-cn.md) | English

OpenAI API compatible Whisper speech recognition service with GPU acceleration and auto-unload.

## Features

- **GPU/CPU Switching** - Support both CUDA and CPU modes
- **Smart Model Management** - Use local models if available, download if not (no update checks)
- **Auto Unload** - Automatically free memory after idle timeout, configurable or disable
- **API Compatible** - Fully compatible with OpenAI `/v1/audio/transcriptions` endpoint
- **Dynamic Model Selection** - Specify model in API request

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

### HF_HUB_OFFLINE Modes

- `auto` - **Smart mode (recommended)**: Use local if exists, download if not
- `1` - Fully offline: Error if not found locally
- `0` - Fully online: Check for updates every time

## API Usage

### Transcription

```bash
curl -X POST http://localhost:5012/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=small" \
  -F "response_format=json"
```

### Supported Models

`tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v2`, `large-v3`

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
  "idle_timeout": 600,
  "hf_offline": "auto"
}
```

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
