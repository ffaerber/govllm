package app

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func cliEnv(t *testing.T) {
	t.Helper()
	t.Setenv("GOVLLM_MODELS_DIR", t.TempDir())
	t.Setenv("HF_ENDPOINT", "https://huggingface.co")
	t.Setenv("VLLM_API_KEY", "")
}

func TestCLIHelpAndValidation(t *testing.T) {
	cliEnv(t)
	for _, args := range [][]string{nil, {"--help"}, {"version"}, {"serve", "--help"}, {"run", "--help"}} {
		var out, logs bytes.Buffer
		if err := Run(context.Background(), args, strings.NewReader(""), &out, &logs); err != nil {
			t.Fatalf("%v: %v", args, err)
		}
		if out.Len()+logs.Len() == 0 {
			t.Fatalf("%v produced no help", args)
		}
	}
	for _, args := range [][]string{{"unknown"}, {"pull"}, {"show"}, {"rm"}, {"list", "extra"}, {"serve", "--backend", "bad"}, {"serve", "--engine-start-timeout", "0"}, {"run"}} {
		if err := Run(context.Background(), args, strings.NewReader(""), io.Discard, io.Discard); err == nil {
			t.Fatalf("invalid command accepted: %v", args)
		}
	}
}

func TestCLIRunAgainstExistingServer(t *testing.T) {
	cliEnv(t)
	backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, fixtureSSE) })
	for _, args := range [][]string{{"run", "--server", backend.URL.String(), "--prompt", "Hi", "test/tiny"}, {"run", "--server", backend.URL.String(), "test/tiny"}} {
		var out bytes.Buffer
		if err := Run(context.Background(), args, strings.NewReader("Hi\n/clear\n/bye\n"), &out, io.Discard); err != nil {
			t.Fatalf("%v: %v", args, err)
		}
		if out.String() != "Hello\n" {
			t.Fatalf("chat output: %q", out.String())
		}
	}
}

func TestCLICacheCommands(t *testing.T) {
	cliEnv(t)
	hub := newHub(t)
	t.Setenv("HF_ENDPOINT", hub.server.URL)
	t.Setenv("HF_TOKEN", "fixture-token")
	t.Setenv("GOVLLM_BACKEND_URL", "http://127.0.0.1:1")
	for _, args := range [][]string{{"pull", "test/tiny"}, {"list"}, {"show", "test/tiny"}, {"rm", "test/tiny"}, {"list"}} {
		if err := Run(context.Background(), args, strings.NewReader(""), io.Discard, io.Discard); err != nil {
			t.Fatalf("%v: %v", args, err)
		}
	}
}

// TestVLLMHelper is a separate OS process used by the supervisor tests. It
// implements only model readiness, not inference; no GPU or Python is needed.
func TestVLLMHelper(t *testing.T) {
	if os.Getenv("GOVLLM_TEST_ENGINE") != "1" {
		return
	}
	if os.Getenv("GOVLLM_TEST_ENGINE_FAIL") == "1" {
		os.Exit(23)
	}
	var port, name string
	for i, arg := range os.Args {
		if i+1 < len(os.Args) {
			if arg == "--port" {
				port = os.Args[i+1]
			}
			if arg == "--served-model-name" {
				name = os.Args[i+1]
			}
		}
	}
	if port == "" || name == "" {
		os.Exit(24)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		if os.Getenv("GOVLLM_TEST_ENGINE_NOT_READY") == "1" {
			http.Error(w, "loading", 503)
			return
		}
		fmt.Fprintf(w, `{"data":[{"id":%q}]}`, name)
	})
	if err := http.ListenAndServe("127.0.0.1:"+port, mux); err != nil {
		os.Exit(25)
	}
	os.Exit(0)
}

func helperEngine(t *testing.T) Config {
	t.Helper()
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// Use a wrapper to preserve vLLM's CLI shape while running the Go fixture.
	wrapper := filepath.Join(t.TempDir(), "vllm-fixture")
	script := "#!/bin/sh\nexec '" + strings.ReplaceAll(binary, "'", "'\\''") + "' -test.run=^TestVLLMHelper$ -- \"$@\"\n"
	if err := os.WriteFile(wrapper, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOVLLM_TEST_ENGINE", "1")
	return Config{VLLMBin: wrapper}
}

func TestManagedEngineReadinessAndStop(t *testing.T) {
	c := helperEngine(t)
	eng, err := startEngine(context.Background(), c, "/fixture/model", "test/tiny", nil, 10*time.Second, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(eng.stop)
	names, err := eng.backend.Models(context.Background())
	if err != nil || len(names) != 1 || names[0] != "test/tiny" {
		t.Fatalf("model readiness: %v %v", names, err)
	}
	eng.stop()
	select {
	case <-eng.done:
	case <-time.After(2 * time.Second):
		t.Fatal("engine not reaped")
	}
}

func TestManagedEngineExitAndTimeout(t *testing.T) {
	c := helperEngine(t)
	t.Setenv("GOVLLM_TEST_ENGINE_FAIL", "1")
	if _, err := startEngine(context.Background(), c, "/fixture/model", "test/tiny", nil, 5*time.Second, io.Discard); err == nil || !strings.Contains(err.Error(), "exited") {
		t.Fatalf("failed engine accepted: %v", err)
	}
	t.Setenv("GOVLLM_TEST_ENGINE_FAIL", "")
	t.Setenv("GOVLLM_TEST_ENGINE_NOT_READY", "1")
	if _, err := startEngine(context.Background(), c, "/fixture/model", "test/tiny", nil, 700*time.Millisecond, io.Discard); err == nil || !strings.Contains(err.Error(), "deadline") {
		t.Fatalf("engine timeout missing: %v", err)
	}
	for _, flag := range []string{"--host", "--port=9000", "--served-model-name", "--api-key=secret"} {
		if _, err := startEngine(context.Background(), c, "/fixture/model", "test/tiny", []string{flag}, time.Second, io.Discard); err == nil {
			t.Fatalf("accepted reserved flag %q", flag)
		}
	}
}

func TestServeStartsAndStopsWithContext(t *testing.T) {
	cliEnv(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{}, 1)
	logs := notifyWriter{ready: started}
	finished := make(chan error, 1)
	go func() {
		finished <- Run(ctx, []string{"serve", "--listen", "127.0.0.1:0", "--backend", "http://127.0.0.1:1"}, strings.NewReader(""), io.Discard, logs)
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("server did not start")
	}
	cancel()
	select {
	case err := <-finished:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("server did not shut down")
	}
}

type notifyWriter struct{ ready chan struct{} }

func (w notifyWriter) Write(p []byte) (int, error) {
	if strings.Contains(string(p), "govllm listening") {
		select {
		case w.ready <- struct{}{}:
		default:
		}
	}
	return len(p), nil
}
