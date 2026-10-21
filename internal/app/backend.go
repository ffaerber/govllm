package app

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type Backend struct {
	URL    *url.URL
	Key    string
	Client *http.Client
}

func NewBackend(raw, key string) (*Backend, error) {
	u, err := baseURL(raw)
	if err != nil {
		return nil, fmt.Errorf("backend: %w", err)
	}
	return &Backend{URL: u, Key: key, Client: &http.Client{
		Transport:     &http.Transport{Proxy: http.ProxyFromEnvironment, ResponseHeaderTimeout: 10 * time.Minute, TLSHandshakeTimeout: 15 * time.Second},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}}, nil
}

func (b *Backend) Request(ctx context.Context, method, path string, body io.Reader) (*http.Response, error) {
	u := *b.URL
	u.Path += path
	req, err := http.NewRequestWithContext(ctx, method, u.String(), body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if b.Key != "" {
		req.Header.Set("Authorization", "Bearer "+b.Key)
	}
	resp, err := b.Client.Do(req)
	if err != nil {
		return nil, errors.New("vLLM is unavailable; start vllm serve or use govllm serve --model MODEL")
	}
	return resp, nil
}

func (b *Backend) Models(ctx context.Context) ([]string, error) {
	resp, err := b.Request(ctx, "GET", "/v1/models", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vLLM model listing returned HTTP %d", resp.StatusCode)
	}
	var data struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&data); err != nil {
		return nil, err
	}
	names := []string{}
	for _, model := range data.Data {
		names = append(names, model.ID)
	}
	return names, nil
}

type engine struct {
	backend *Backend
	stop    func()
	done    <-chan struct{}
}

// Start one child engine, wait for real model readiness, and reap it on every exit.
// Arguments are passed directly to exec: model names and flags never enter a shell.
func startEngine(ctx context.Context, c Config, modelPath, name string, extra []string, timeout time.Duration, logs io.Writer) (*engine, error) {
	bin, err := exec.LookPath(c.VLLMBin)
	if err != nil {
		return nil, fmt.Errorf("vLLM executable %q not found; use the Docker image or install vLLM for your GPU", c.VLLMBin)
	}
	for _, arg := range extra {
		for _, reserved := range []string{"--host", "--port", "--served-model-name", "--model", "--api-key"} {
			if arg == reserved || strings.HasPrefix(arg, reserved+"=") {
				return nil, fmt.Errorf("%s is managed by govllm; use VLLM_API_KEY for authentication", reserved)
			}
		}
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	backend, err := NewBackend("http://127.0.0.1:"+strconv.Itoa(port), c.BackendKey)
	if err != nil {
		return nil, err
	}
	args := []string{"serve", modelPath, "--host", "127.0.0.1", "--port", strconv.Itoa(port), "--served-model-name", name}
	args = append(args, extra...)
	cmd := exec.Command(bin, args...)
	cmd.Stdout, cmd.Stderr = logs, logs
	// vLLM spawns workers; terminate the process group, not just its launcher.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	done := make(chan struct{})
	var waitErr error
	go func() { waitErr = cmd.Wait(); close(done) }()
	stop := func() {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			<-done
		}
	}
	readyCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			stop()
			return nil, fmt.Errorf("vLLM exited before readiness: %v (see engine logs)", waitErr)
		case <-readyCtx.Done():
			stop()
			return nil, fmt.Errorf("waiting for vLLM: %w", readyCtx.Err())
		case <-ticker.C:
			probe, cancel := context.WithTimeout(readyCtx, 2*time.Second)
			names, err := backend.Models(probe)
			cancel()
			if err == nil {
				for _, n := range names {
					if n == name {
						return &engine{backend: backend, stop: stop, done: done}, nil
					}
				}
			}
		}
	}
}

type Message struct {
	Role      string          `json:"role"`
	Content   string          `json:"content"`
	Images    []string        `json:"images,omitempty"`
	ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
}

type completionChunk struct {
	Choices []struct {
		Text  string `json:"text"`
		Delta struct {
			Content  string `json:"content"`
			Thinking string `json:"reasoning_content"`
		} `json:"delta"`
		Message struct {
			Content  string `json:"content"`
			Thinking string `json:"reasoning_content"`
		} `json:"message"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		Prompt     int `json:"prompt_tokens"`
		Completion int `json:"completion_tokens"`
	} `json:"usage"`
	Error json.RawMessage `json:"error"`
}

// vLLM emits one JSON value per SSE event. Require [DONE] so dropped connections
// cannot be mistaken for successful generations.
func consumeSSE(reader io.Reader, consume func(completionChunk) error) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64<<10), 8<<20)
	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return nil
		}
		if data == "" {
			continue
		}
		var chunk completionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return errors.New("invalid JSON in vLLM stream")
		}
		if len(chunk.Error) > 0 {
			return errors.New("vLLM reported an error during generation")
		}
		if err := consume(chunk); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return errors.New("vLLM stream ended before [DONE]")
}

func postJSON(ctx context.Context, b *Backend, endpoint string, body any) (*http.Response, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	return b.Request(ctx, http.MethodPost, endpoint, bytes.NewReader(data))
}

// resolveModel uses a downloaded snapshot or an explicit local directory. Local
// paths are accepted only at the CLI, never from remote API clients.
func resolveModel(ctx context.Context, manager *Manager, name string, logs io.Writer) (string, string, error) {
	if strings.HasPrefix(name, "/") || strings.HasPrefix(name, ".") {
		info, err := os.Stat(name)
		if err != nil || !info.IsDir() {
			return "", "", errors.New("local model path must be an existing directory")
		}
		return name, name, nil
	}
	model, err := manager.Show(name)
	if errors.Is(err, ErrNotFound) {
		model, err = manager.Pull(ctx, name, "main", func(p Progress) { fmt.Fprintln(logs, p.Status) })
	}
	if err != nil {
		return "", "", err
	}
	return model.Path, name, nil
}
