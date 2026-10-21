# govllm

An Ollama-style CLI and API written in Go, using **vLLM for inference**. Inspired
by [Ovllm](https://github.com/FearL0rd/Ovllm), with a Go control layer and the
official `vllm/vllm-openai` image as the Docker runtime base.

Go handles model downloads, local storage, HTTP compatibility, interactive chat,
and the vLLM process lifecycle. vLLM retains its Python, PyTorch, CUDA,
continuous batching, and GPU support. This is an initial implementation, not a
port of the inference engine or full feature parity with Ollama/Ovllm.

## Docker quick start

Requires Linux, Docker Compose with GPU support, NVIDIA Container Toolkit, and
a GPU/driver supported by the selected vLLM image. Choose a model that fits
your GPU. The image and model weights can require tens of GB of disk space.

```bash
docker compose up --build -d
docker compose logs -f govllm
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. The first startup downloads
and verifies its Transformers files, then starts vLLM privately on loopback.
The public Go API starts once vLLM reports the expected model as loaded.

```bash
curl --fail http://127.0.0.1:11434/ready
curl http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello!"}],"stream":false}'
```

Model storage persists in the `models` volume. Set `MODEL` to choose a different
Hugging Face model. For gated/private models, set `HF_TOKEN` in your process
environment and accept the model license on Hugging Face first; do not commit
tokens. Both public and gated models need access to Hugging Face and its
download destinations.

The Dockerfile builds a static Go binary in `golang:1.27.3-bookworm` and copies
it into the **official vLLM image**, pinned by manifest digest. It replaces the
upstream entrypoint with `govllm`, which supervises `vllm serve`. Override the
base to select another vLLM/CUDA release:

```bash
VLLM_IMAGE=vllm/vllm-openai:YOUR_VERSION docker compose build
# Or:
docker build --build-arg VLLM_IMAGE=vllm/vllm-openai:YOUR_VERSION -t govllm:local .
```

For a manual container with vLLM tuning:

```bash
docker build -t govllm:local .
docker run --rm --gpus all --ipc=host \
  -p 127.0.0.1:11434:11434 -v govllm-models:/models \
  govllm:local serve --model Qwen/Qwen2.5-0.5B-Instruct -- \
  --gpu-memory-utilization 0.8 --max-model-len 4096
```

`--host`, `--port`, `--model`, `--served-model-name`, and `--api-key` belong to
govllm's supervisor. Use `VLLM_API_KEY` for private backend authentication. Other
vLLM flags pass through as arguments without a shell. Python model code is not
downloaded or trusted automatically; models requiring custom Python code are
outside the supported download workflow.

## Native Go workflow

Go 1.24+ on Linux is sufficient to build and test the Go application; there are
no third-party Go dependencies. A local vLLM installation and compatible GPU
are needed only when starting a managed inference engine. You can instead use
an existing vLLM server, including one on another GPU machine.

```bash
go build -o bin/govllm ./cmd/govllm
go test -race ./...
go vet ./...

# Pull and manage Transformers models without starting vLLM:
bin/govllm pull Qwen/Qwen2.5-0.5B-Instruct
bin/govllm list
bin/govllm show Qwen/Qwen2.5-0.5B-Instruct

# Pull if needed, start vLLM, and chat; /bye exits, /clear resets history:
bin/govllm run Qwen/Qwen2.5-0.5B-Instruct

# Start the Go API and a managed vLLM process:
bin/govllm serve --model Qwen/Qwen2.5-0.5B-Instruct

# Or connect the Go API to vLLM that is already running:
bin/govllm serve --backend http://GPU_HOST:8000

# Chat through that existing API without launching another engine:
bin/govllm run --server http://127.0.0.1:11434 Qwen/Qwen2.5-0.5B-Instruct
bin/govllm run --server http://127.0.0.1:11434 --prompt 'Hello' Qwen/Qwen2.5-0.5B-Instruct

bin/govllm ps --backend http://127.0.0.1:11434
```

CLI flags precede positional model names. `serve --model /absolute/path` and
`run /absolute/path` accept local Transformers directories. Managed startup
waits up to ten minutes; set `--engine-start-timeout 20m` for slower systems.
Stop the engine before `bin/govllm rm OWNER/MODEL`; also stop any other process
using that cache. The CLI checks the configured backend but cannot discover
unrelated engines. The HTTP deletion endpoint refuses to delete loaded models.

## APIs and OpenWebUI

Configure OpenWebUI's **Ollama connection** to the address of govllm as reachable
from OpenWebUI's server. If both containers share a Docker network, use the
govllm service name and port 11434. API clients need to select the model already
loaded by vLLM. Pulling adds a local model; it does not load or switch engines.
Restart `serve --model` to change the active model.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Go process liveness |
| `GET /ready` | vLLM reachable with at least one loaded model |
| `GET /api/version` | Go application version |
| `GET /api/tags` | Local models plus models served by the configured backend |
| `GET /api/ps` | Models reported by vLLM |
| `POST /api/show` | Basic model metadata and completion capability |
| `POST /api/pull` | Verified download; NDJSON progress by default |
| `DELETE /api/delete` | Remove an unused local model |
| `POST /api/chat` | Ollama text chat; NDJSON streaming or JSON |
| `POST /api/generate` | Ollama text generation using vLLM's chat template |
| `GET /v1/models` | Proxy vLLM model listing |
| `POST /v1/chat/completions` | Proxy vLLM, including SSE streaming |
| `POST /v1/completions` | Proxy vLLM text completions |

The `/v1` routes preserve vLLM request/response semantics, including errors and
streaming. Use them for advanced engine features. For example, point an OpenAI
client at govllm's `/v1` base URL and use the served model name.

The Ollama adapter supports text messages, system prompts on `/api/generate`,
`format: "json"` or a JSON schema, and common sampling options (`temperature`,
`top_p`, `top_k`, `num_predict`, `stop`, `seed`, `repeat_penalty`). Zero-valued
options are preserved. Usage comes from the engine. `stream` defaults to `true`.
Images, tools, raw prompts, custom templates, and token contexts are rejected
by the adapter. Ollama Modelfiles, embeddings, dynamic model switching, GGUF
quantization suffixes, split-GGUF merging, and OpenWebUI end-to-end validation
are not implemented in this version. `/api/show` returns basic metadata; GPU
memory and parameter counts are not fabricated.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOVLLM_ADDR` | `127.0.0.1:11434` | API listen address (`0.0.0.0:11434` inside Docker) |
| `GOVLLM_MODELS_DIR` | `~/.govllm/models` | Snapshot and manifest storage (`/models` in Docker) |
| `GOVLLM_BACKEND_URL` | `http://127.0.0.1:8000` | Existing vLLM base URL; optional `/v1` suffix accepted |
| `GOVLLM_VLLM_BIN` | `vllm` | Managed engine executable |
| `VLLM_API_KEY` | unset | Authentication sent only to the configured vLLM backend |
| `HF_ENDPOINT` | `https://huggingface.co` | Hub endpoint or trusted compatible mirror |
| `HF_TOKEN` | unset | Authentication for the selected Hub |

The Go API has no client authentication. Native mode and Compose publish to
loopback by default. Use an authenticated reverse proxy for shared deployments.
Do not expose the model download/delete API to untrusted clients.

Downloads pin a Hub commit and verify each file using its Git blob or LFS
SHA-256 digest. Safetensors take priority over duplicate PyTorch `.bin` weights.
Only root-level Transformers weights, tokenizer files, and configuration are
selected. A manifest becomes visible after the entire download succeeds;
cancellation/checksum failures remove staging files. Refresh with
`govllm pull --revision REV OWNER/MODEL`. Older revision snapshots stay available
until the model is removed. A filesystem lock prevents concurrent cache
mutations by separate processes. If a process is killed without cleanup,
confirm no model operation is running before removing `.mutation-lock` from
the model directory.

## Validation

Tests run without a GPU and use local HTTP fixtures for Hub downloads and the
vLLM API. They exercise verified downloads, cache reuse/deletion, corrupt and
unsafe manifests, cancellation, token redirect isolation, SSE/NDJSON conversion,
usage/sampling preservation, HTTP errors, CLI workflows, and child-process
readiness/shutdown. They do not prove CUDA inference or complete OpenWebUI
compatibility; validate those on the intended GPU host with a real model.

Licensed under Apache-2.0. The Go implementation is new code; Ovllm supplies the
workflow inspiration and vLLM supplies inference.
