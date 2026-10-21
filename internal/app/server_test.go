package app

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func fakeBackend(t *testing.T, handler http.HandlerFunc) *Backend {
	t.Helper()
	upstream := httptest.NewServer(handler)
	t.Cleanup(upstream.Close)
	b, err := NewBackend(upstream.URL, "engine-key")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func testHandler(t *testing.T, backend *Backend) http.Handler {
	t.Helper()
	m, err := NewManager(Config{ModelsDir: t.TempDir(), HFEndpoint: "https://huggingface.co"})
	if err != nil {
		t.Fatal(err)
	}
	return NewHandler(m, backend)
}

func request(t *testing.T, handler http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	handler.ServeHTTP(rec, req)
	return rec
}

func TestOllamaChatPreservesSamplingAndUsage(t *testing.T) {
	backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer engine-key" {
			t.Error("wrong upstream route or key")
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["temperature"] != float64(0) || body["max_tokens"] != float64(17) || body["stream"] != false {
			t.Errorf("sampling changed: %v", body)
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}`)
	})
	rec := request(t, testHandler(t, backend), "POST", "/api/chat", `{"model":"test/tiny","messages":[{"role":"user","content":"Hi"}],"stream":false,"options":{"temperature":0,"num_predict":17}}`)
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body.String())
	}
	var response struct {
		Message Message `json:"message"`
		Done    bool    `json:"done"`
		Prompt  int     `json:"prompt_eval_count"`
		Eval    int     `json:"eval_count"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Message.Content != "Hello" || !response.Done || response.Prompt != 4 || response.Eval != 2 {
		t.Fatalf("bad response: %+v", response)
	}
}

func TestOllamaGenerateAppliesSystemAndSchema(t *testing.T) {
	backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []Message      `json:"messages"`
			Format   map[string]any `json:"response_format"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		want := []Message{{Role: "system", Content: "Be concise"}, {Role: "user", Content: "Hi"}}
		if !reflect.DeepEqual(body.Messages, want) || body.Format["type"] != "json_object" {
			t.Errorf("bad translation: %+v", body)
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}`)
	})
	rec := request(t, testHandler(t, backend), "POST", "/api/generate", `{"model":"test/tiny","system":"Be concise","prompt":"Hi","format":"json","stream":false}`)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"response":"{}"`) {
		t.Fatalf("%d %s", rec.Code, rec.Body.String())
	}
}

const fixtureSSE = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" +
	"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n" +
	"data: [DONE]\n\n"

func TestOllamaStreamingNDJSON(t *testing.T) {
	backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, fixtureSSE)
	})
	for _, route := range []string{"chat", "generate"} {
		t.Run(route, func(t *testing.T) {
			rec := request(t, testHandler(t, backend), "POST", "/api/"+route, `{"model":"test/tiny","prompt":"Hi","messages":[{"role":"user","content":"Hi"}]}`)
			if rec.Header().Get("Content-Type") != "application/x-ndjson" || !rec.Flushed {
				t.Fatalf("stream not flushed: %v", rec.Header())
			}
			scanner := bufio.NewScanner(rec.Body)
			count := 0
			text := ""
			for scanner.Scan() {
				var body map[string]any
				if err := json.Unmarshal(scanner.Bytes(), &body); err != nil {
					t.Fatal(err)
				}
				count++
				if body["done"] == true {
					if count != 3 || body["done_reason"] != "length" || body["eval_count"] != float64(2) {
						t.Fatalf("bad final chunk: %v", body)
					}
				} else {
					if route == "chat" {
						text += body["message"].(map[string]any)["content"].(string)
					} else {
						text += body["response"].(string)
					}
				}
			}
			if count != 3 || text != "Hello" {
				t.Fatalf("count=%d text=%q", count, text)
			}
		})
	}
}

func TestTruncatedStreamReportsErrorNotDone(t *testing.T) {
	backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, strings.TrimSuffix(fixtureSSE, "data: [DONE]\n\n"))
	})
	rec := request(t, testHandler(t, backend), "POST", "/api/generate", `{"model":"test/tiny","prompt":"Hi"}`)
	if !strings.Contains(rec.Body.String(), `"error":`) || strings.Contains(rec.Body.String(), `"done":true`) {
		t.Fatalf("false success: %s", rec.Body.String())
	}
}

func TestOpenAIProxyPreservesStreamAndErrors(t *testing.T) {
	for _, status := range []int{200, 429} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer engine-key" {
					t.Error("client key forwarded instead of engine key")
				}
				body, _ := io.ReadAll(r.Body)
				if !bytes.Equal(body, []byte(`{"model":"test/tiny","stream":true}`)) {
					t.Errorf("body changed: %s", body)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(status)
				fmt.Fprint(w, fixtureSSE)
			})
			handler := testHandler(t, backend)
			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"test/tiny","stream":true}`))
			req.Header.Set("Authorization", "Bearer caller-key")
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != status || rec.Body.String() != fixtureSSE {
				t.Fatalf("proxy changed response: %d %s", rec.Code, rec.Body.String())
			}
		})
	}
}

func TestTagsIncludeExternalModelsAndReadiness(t *testing.T) {
	backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, `{"data":[{"id":"external/model"}]}`) })
	handler := testHandler(t, backend)
	for _, route := range []string{"/api/tags", "/api/ps", "/ready"} {
		rec := request(t, handler, "GET", route, "")
		if rec.Code != 200 || !strings.Contains(rec.Body.String(), "external/model") {
			t.Fatalf("%s: %d %s", route, rec.Code, rec.Body.String())
		}
	}
	rec := request(t, handler, "POST", "/api/show", `{"model":"external/model"}`)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"capabilities":["completion"]`) {
		t.Fatalf("show: %d %s", rec.Code, rec.Body.String())
	}
}

func TestModelAPIManagementLifecycle(t *testing.T) {
	hub := newHub(t)
	m := managerFor(t, hub)
	backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, `{"data":[]}`) })
	handler := NewHandler(m, backend)
	for _, stream := range []string{"false", "true"} {
		rec := request(t, handler, "POST", "/api/pull", `{"model":"test/tiny","stream":`+stream+`}`)
		if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"status":"success"`) {
			t.Fatalf("pull: %d %s", rec.Code, rec.Body.String())
		}
	}
	if rec := request(t, handler, "GET", "/api/tags", ""); !strings.Contains(rec.Body.String(), "test/tiny") {
		t.Fatal("pulled model missing from tags")
	}
	if rec := request(t, handler, "DELETE", "/api/delete", `{"model":"test/tiny"}`); rec.Code != 200 {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body.String())
	}
	if rec := request(t, handler, "GET", "/ready", ""); rec.Code != 503 {
		t.Fatal("empty backend ready")
	}
}

func TestBadRequestsAndBackendFailures(t *testing.T) {
	backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) { http.Error(w, "model not found", 404) })
	handler := testHandler(t, backend)
	for _, body := range []string{
		`{`, `{}`, `{} {}`, `{"model":"test/tiny","messages":[]}`, `{"model":"test/tiny","messages":[{"role":"user","content":"Hi","images":["base64"]}]}`,
		`{"model":"test/tiny","messages":[{"role":"user","content":"Hi"}],"format":"xml"}`,
		`{"model":"test/tiny","messages":[{"role":"user","content":"Hi"}],"tools":[{}]}`,
		`{"model":"test/tiny","messages":[{"role":"user","content":"` + strings.Repeat("x", 1<<20) + `"}]}`,
	} {
		if rec := request(t, handler, "POST", "/api/chat", body); rec.Code != 400 {
			t.Fatalf("accepted invalid request: %d %s", rec.Code, rec.Body.String())
		}
	}
	if rec := request(t, handler, "POST", "/api/chat", `{"model":"test/tiny","messages":[{"role":"user","content":"Hi"}],"stream":false}`); rec.Code != 404 {
		t.Fatalf("lost upstream error status: %d", rec.Code)
	}
	if rec := request(t, handler, "GET", "/ready", ""); rec.Code != 503 {
		t.Fatal("unhealthy backend reported ready")
	}
}

func TestGenerationCancellationReachesBackend(t *testing.T) {
	canceled := make(chan struct{})
	started := make(chan struct{})
	backend := fakeBackend(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		close(started)
		<-r.Context().Done()
		close(canceled)
	})
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest("POST", "/api/generate", strings.NewReader(`{"model":"test/tiny","prompt":"Hi"}`)).WithContext(ctx)
	finished := make(chan struct{})
	handler := testHandler(t, backend)
	go func() { handler.ServeHTTP(httptest.NewRecorder(), req); close(finished) }()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("backend never called")
	}
	cancel()
	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("backend context not canceled")
	}
	select {
	case <-finished:
	case <-time.After(3 * time.Second):
		t.Fatal("handler did not stop")
	}
}
