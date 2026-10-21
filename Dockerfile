# syntax=docker/dockerfile:1
ARG GO_IMAGE=golang:1.27.3-bookworm
ARG VLLM_IMAGE=vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

FROM ${GO_IMAGE} AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/govllm ./cmd/govllm

# Keep the official vLLM CUDA, PyTorch, Python, and engine installation intact.
FROM ${VLLM_IMAGE}
COPY --from=build /out/govllm /usr/local/bin/govllm
ENV GOVLLM_ADDR=0.0.0.0:11434 GOVLLM_MODELS_DIR=/models
VOLUME ["/models"]
EXPOSE 11434
HEALTHCHECK --start-period=10m --interval=30s --timeout=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/ready', timeout=4)" || exit 1
# Replace the upstream OpenAI-server entrypoint: govllm supervises vLLM itself.
ENTRYPOINT ["/usr/local/bin/govllm"]
CMD ["serve", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]
