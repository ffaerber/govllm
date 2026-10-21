package app

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"strings"
	"text/tabwriter"
	"time"
)

const help = `govllm — Ollama-style Go CLI/API backed by vLLM

Usage:
  govllm pull [--revision main] OWNER/MODEL
  govllm list
  govllm show OWNER/MODEL
  govllm rm OWNER/MODEL
  govllm ps [--backend http://127.0.0.1:8000]
  govllm serve [--listen 127.0.0.1:11434] [--backend URL]
  govllm serve --model OWNER/MODEL [-- VLLM_FLAGS...]
  govllm run [--prompt TEXT] [--server URL] MODEL [-- VLLM_FLAGS...]
  govllm version

Flags go before the model argument. With no --server, run pulls the model,
starts a private vLLM engine, and opens a chat (/bye to exit, /clear to reset).
With --server, run connects to an existing govllm/vLLM API using its loaded model.
Use serve --model /absolute/model/path for a local Transformers model.

Environment:
  GOVLLM_ADDR, GOVLLM_MODELS_DIR, GOVLLM_BACKEND_URL, GOVLLM_VLLM_BIN
  HF_TOKEN, HF_ENDPOINT, VLLM_API_KEY
`

func Run(ctx context.Context, args []string, in io.Reader, out, logs io.Writer) error {
	if len(args) == 0 || args[0] == "--help" || args[0] == "help" || args[0] == "-h" {
		_, err := io.WriteString(out, help)
		return err
	}
	if args[0] == "version" || args[0] == "--version" {
		fmt.Fprintln(out, Version)
		return nil
	}
	c, err := LoadConfig()
	if err != nil {
		return err
	}
	manager, err := NewManager(c)
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	fs.SetOutput(logs)
	switch args[0] {
	case "pull":
		revision := fs.String("revision", "main", "Hugging Face revision")
		if err := fs.Parse(args[1:]); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil
			}
			return err
		}
		if fs.NArg() != 1 {
			return errors.New("usage: govllm pull [--revision REV] OWNER/MODEL")
		}
		model, err := manager.Pull(ctx, fs.Arg(0), *revision, func(p Progress) { fmt.Fprintln(logs, p.Status) })
		if err != nil {
			return err
		}
		fmt.Fprintln(out, model.Path)
		return nil
	case "list":
		if len(args) != 1 {
			return errors.New("usage: govllm list")
		}
		models, err := manager.List()
		if err != nil {
			return err
		}
		w := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
		fmt.Fprintln(w, "NAME\tBYTES\tREVISION\tUPDATED")
		for _, m := range models {
			fmt.Fprintf(w, "%s\t%d\t%s\t%s\n", m.Name, m.Size, m.Digest[:12], m.ModifiedAt.Format(time.RFC3339))
		}
		return w.Flush()
	case "show":
		if len(args) != 2 {
			return errors.New("usage: govllm show OWNER/MODEL")
		}
		model, err := manager.Show(args[1])
		if err != nil {
			return err
		}
		e := json.NewEncoder(out)
		e.SetIndent("", "  ")
		return e.Encode(model)
	case "rm":
		if len(args) != 2 {
			return errors.New("usage: govllm rm OWNER/MODEL")
		}
		backend, err := NewBackend(c.BackendURL, c.BackendKey)
		if err != nil {
			return err
		}
		probe, cancel := shortContext(ctx)
		names, backendErr := backend.Models(probe)
		cancel()
		if backendErr == nil {
			for _, name := range names {
				if name == args[1] {
					return errors.New("model is in use by the configured vLLM backend; stop it before removing")
				}
			}
		}
		if err := manager.Remove(args[1]); err != nil {
			return err
		}
		fmt.Fprintln(out, "removed", args[1])
		return nil
	case "ps":
		backendURL := fs.String("backend", c.BackendURL, "vLLM base URL")
		if err := fs.Parse(args[1:]); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil
			}
			return err
		}
		if fs.NArg() != 0 {
			return errors.New("usage: govllm ps [--backend URL]")
		}
		backend, err := NewBackend(*backendURL, c.BackendKey)
		if err != nil {
			return err
		}
		probe, cancel := shortContext(ctx)
		defer cancel()
		names, err := backend.Models(probe)
		if err != nil {
			return err
		}
		for _, name := range names {
			fmt.Fprintln(out, name)
		}
		return nil
	case "serve":
		addr := fs.String("listen", c.Address, "HTTP listen address")
		backendURL := fs.String("backend", c.BackendURL, "existing vLLM base URL")
		model := fs.String("model", "", "model to pull and serve with a managed vLLM engine")
		timeout := fs.Duration("engine-start-timeout", 10*time.Minute, "maximum engine startup time")
		if err := fs.Parse(args[1:]); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil
			}
			return err
		}
		if *timeout <= 0 {
			return errors.New("engine-start-timeout must be positive")
		}
		if *model == "" && fs.NArg() > 0 {
			return errors.New("vLLM flags require --model")
		}
		if *model != "" {
			conflict := false
			fs.Visit(func(f *flag.Flag) {
				if f.Name == "backend" {
					conflict = true
				}
			})
			if conflict {
				return errors.New("choose --model or --backend")
			}
		}
		backend, err := NewBackend(*backendURL, c.BackendKey)
		if err != nil {
			return err
		}
		// Reserve the public port before a potentially expensive model download.
		listener, err := net.Listen("tcp", *addr)
		if err != nil {
			return err
		}
		defer listener.Close()
		var eng *engine
		if *model != "" {
			if _, err := exec.LookPath(c.VLLMBin); err != nil {
				return fmt.Errorf("vLLM is not installed; use the Docker image or --backend URL")
			}
			path, name, err := resolveModel(ctx, manager, *model, logs)
			if err != nil {
				return err
			}
			eng, err = startEngine(ctx, c, path, name, fs.Args(), *timeout, logs)
			if err != nil {
				return err
			}
			defer eng.stop()
			backend = eng.backend
		}
		server := &http.Server{Handler: NewHandler(manager, backend), ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 2 * time.Minute, BaseContext: func(net.Listener) context.Context { return ctx }}
		finished := make(chan error, 1)
		go func() { finished <- server.Serve(listener) }()
		fmt.Fprintf(logs, "govllm listening on %s; vLLM backend %s\n", listener.Addr(), backend.URL.Redacted())
		var engineDone <-chan struct{}
		if eng != nil {
			engineDone = eng.done
		}
		var result error
		select {
		case <-ctx.Done():
		case <-engineDone:
			result = errors.New("vLLM engine exited; see engine logs")
		case err := <-finished:
			if !errors.Is(err, http.ErrServerClosed) {
				result = err
			}
		}
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdown); err != nil {
			_ = server.Close()
		}
		return result
	case "run":
		serverURL := fs.String("server", "", "existing govllm/vLLM base URL")
		prompt := fs.String("prompt", "", "one-shot prompt (otherwise interactive)")
		timeout := fs.Duration("engine-start-timeout", 10*time.Minute, "maximum engine startup time")
		if err := fs.Parse(args[1:]); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil
			}
			return err
		}
		if fs.NArg() == 0 {
			return errors.New("usage: govllm run [--prompt TEXT] [--server URL] MODEL [-- VLLM_FLAGS...]")
		}
		if *timeout <= 0 {
			return errors.New("engine-start-timeout must be positive")
		}
		name := fs.Arg(0)
		extra := fs.Args()[1:]
		if len(extra) > 0 && extra[0] == "--" {
			extra = extra[1:]
		}
		var backend *Backend
		if *serverURL != "" {
			if len(extra) > 0 {
				return errors.New("vLLM flags cannot be used with --server")
			}
			backend, err = NewBackend(*serverURL, c.BackendKey)
			if err != nil {
				return err
			}
		} else {
			if _, err := exec.LookPath(c.VLLMBin); err != nil {
				return errors.New("vLLM is not installed; use the Docker image or run --server URL MODEL")
			}
			path, served, err := resolveModel(ctx, manager, name, logs)
			if err != nil {
				return err
			}
			name = served
			eng, err := startEngine(ctx, c, path, name, extra, *timeout, logs)
			if err != nil {
				return err
			}
			defer eng.stop()
			backend = eng.backend
		}
		if *prompt != "" {
			_, err := chatOnce(ctx, backend, name, []Message{{Role: "user", Content: *prompt}}, out)
			return err
		}
		return interactive(ctx, backend, name, in, out, logs)
	default:
		return fmt.Errorf("unknown command %q; see govllm --help", args[0])
	}
}

func chatOnce(ctx context.Context, backend *Backend, model string, messages []Message, out io.Writer) (string, error) {
	resp, err := postJSON(ctx, backend, "/v1/chat/completions", map[string]any{"model": model, "messages": messages, "stream": true})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("chat returned HTTP %d; verify that %q is loaded by the backend", resp.StatusCode, model)
	}
	var answer strings.Builder
	err = consumeSSE(resp.Body, func(chunk completionChunk) error {
		if len(chunk.Choices) == 0 {
			return nil
		}
		text := chunk.Choices[0].Delta.Content
		answer.WriteString(text)
		_, err := io.WriteString(out, text)
		return err
	})
	if err != nil {
		return "", err
	}
	_, err = fmt.Fprintln(out)
	return answer.String(), err
}

func interactive(ctx context.Context, backend *Backend, model string, in io.Reader, out, logs io.Writer) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	fmt.Fprintln(logs, "Chat with", model, "(/bye to exit, /clear to reset)")
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	type line struct {
		text string
		err  error
		eof  bool
	}
	lines := make(chan line)
	go func() {
		defer close(lines)
		for scanner.Scan() {
			select {
			case lines <- line{text: scanner.Text()}:
			case <-ctx.Done():
				return
			}
		}
		select {
		case lines <- line{err: scanner.Err(), eof: true}:
		case <-ctx.Done():
		}
	}()
	messages := []Message{}
	for {
		fmt.Fprint(logs, ">>> ")
		var input line
		select {
		case <-ctx.Done():
			return nil
		case input = <-lines:
		}
		if input.eof {
			return input.err
		}
		text := strings.TrimSpace(input.text)
		if text == "/bye" {
			return nil
		}
		if text == "/clear" {
			messages = nil
			continue
		}
		if text == "" {
			continue
		}
		messages = append(messages, Message{Role: "user", Content: text})
		answer, err := chatOnce(ctx, backend, model, messages, out)
		if err != nil {
			return err
		}
		messages = append(messages, Message{Role: "assistant", Content: answer})
	}
}
