.PHONY: build test vet
build:
	go build -trimpath -o bin/govllm ./cmd/govllm
test:
	go test -race ./...
vet:
	go vet ./...
