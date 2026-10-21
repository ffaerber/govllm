package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"time"
)

type Server struct {
	manager *Manager
	backend *Backend
}

func shortContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, 3*time.Second)
}

func NewHandler(manager *Manager, backend *Backend) http.Handler {
	s := &Server{manager: manager, backend: backend}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "govllm is running") })
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("GET /ready", s.ready)
	mux.HandleFunc("GET /api/version", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, map[string]string{"version": Version}) })
	mux.HandleFunc("GET /api/tags", s.tags)
	mux.HandleFunc("GET /api/ps", s.ps)
	mux.HandleFunc("POST /api/show", s.show)
	mux.HandleFunc("POST /api/pull", s.pull)
	mux.HandleFunc("DELETE /api/delete", s.remove)
	mux.HandleFunc("POST /api/chat", func(w http.ResponseWriter, r *http.Request) { s.generate(w, r, true) })
	mux.HandleFunc("POST /api/generate", func(w http.ResponseWriter, r *http.Request) { s.generate(w, r, false) })
	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(backend.URL)
			pr.Out.Header.Del("Authorization")
			if backend.Key != "" {
				pr.Out.Header.Set("Authorization", "Bearer "+backend.Key)
			}
		},
		Transport:     backend.Client.Transport,
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			apiError(w, 502, errors.New("vLLM is unavailable"))
		},
	}
	mux.Handle("GET /v1/models", proxy)
	mux.Handle("POST /v1/chat/completions", proxy)
	mux.Handle("POST /v1/completions", proxy)
	return mux
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
func apiError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}
func readJSON(w http.ResponseWriter, r *http.Request, value any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	d := json.NewDecoder(r.Body)
	if err := d.Decode(value); err != nil {
		return errors.New("invalid JSON body (maximum 1 MiB)")
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return errors.New("expected exactly one JSON object")
	}
	return nil
}

type modelRequest struct {
	Model    string `json:"model"`
	Name     string `json:"name"`
	Revision string `json:"revision"`
	Stream   *bool  `json:"stream"`
}

func (r modelRequest) model() string {
	if r.Model != "" {
		return r.Model
	}
	return r.Name
}
func (r modelRequest) streaming() bool { return r.Stream == nil || *r.Stream }

func (s *Server) ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := shortContext(r.Context())
	defer cancel()
	names, err := s.backend.Models(ctx)
	if err != nil {
		apiError(w, 503, err)
		return
	}
	if len(names) == 0 {
		apiError(w, 503, errors.New("vLLM has no loaded model"))
		return
	}
	writeJSON(w, 200, map[string]any{"status": "ready", "models": names})
}

func (s *Server) tags(w http.ResponseWriter, r *http.Request) {
	models, err := s.manager.List()
	if err != nil {
		apiError(w, 500, err)
		return
	}
	items := []any{}
	seen := map[string]bool{}
	for _, model := range models {
		items = append(items, ollamaModel(model))
		seen[model.Name] = true
	}
	// Externally hosted models must be selectable in OpenWebUI even if the Go
	// model cache does not contain their weights.
	ctx, cancel := shortContext(r.Context())
	defer cancel()
	if names, err := s.backend.Models(ctx); err == nil {
		for _, name := range names {
			if !seen[name] {
				items = append(items, ollamaModel(Model{Name: name}))
			}
		}
	}
	writeJSON(w, 200, map[string]any{"models": items})
}

func ollamaModel(m Model) map[string]any {
	return map[string]any{"name": m.Name, "model": m.Name, "modified_at": m.ModifiedAt, "size": m.Size, "digest": m.Digest, "details": map[string]any{"format": "safetensors", "family": "", "families": []string{}, "parameter_size": "", "quantization_level": ""}}
}

func (s *Server) ps(w http.ResponseWriter, r *http.Request) {
	names, err := s.backend.Models(r.Context())
	if err != nil {
		apiError(w, 502, err)
		return
	}
	models := []any{}
	for _, name := range names {
		model, err := s.manager.Show(name)
		if err != nil {
			model = Model{Name: name}
		}
		models = append(models, ollamaModel(model))
	}
	writeJSON(w, 200, map[string]any{"models": models})
}

func (s *Server) show(w http.ResponseWriter, r *http.Request) {
	var req modelRequest
	if err := readJSON(w, r, &req); err != nil {
		apiError(w, 400, err)
		return
	}
	if req.model() == "" {
		apiError(w, 400, errors.New("model is required"))
		return
	}
	model, err := s.manager.Show(req.model())
	if err != nil {
		names, backendErr := s.backend.Models(r.Context())
		for _, name := range names {
			if name == req.model() {
				model = Model{Name: name}
				err = nil
				break
			}
		}
		if err != nil {
			if backendErr != nil && !errors.Is(err, ErrNotFound) {
				apiError(w, 400, err)
			} else {
				apiError(w, 404, err)
			}
			return
		}
	}
	writeJSON(w, 200, map[string]any{"details": ollamaModel(model)["details"], "model_info": map[string]any{"general.name": model.Name}, "capabilities": []string{"completion"}, "parameters": "", "template": "", "modified_at": model.ModifiedAt})
}

func (s *Server) pull(w http.ResponseWriter, r *http.Request) {
	var req modelRequest
	if err := readJSON(w, r, &req); err != nil {
		apiError(w, 400, err)
		return
	}
	if err := validModel(req.model()); err != nil {
		apiError(w, 400, err)
		return
	}
	if !req.streaming() {
		_, err := s.manager.Pull(r.Context(), req.model(), req.Revision, nil)
		if err != nil {
			apiError(w, 502, err)
			return
		}
		writeJSON(w, 200, Progress{Status: "success"})
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	encoder := json.NewEncoder(w)
	_, err := s.manager.Pull(r.Context(), req.model(), req.Revision, func(p Progress) { _ = encoder.Encode(p); _ = http.NewResponseController(w).Flush() })
	if err != nil {
		_ = encoder.Encode(map[string]string{"error": err.Error()})
		_ = http.NewResponseController(w).Flush()
	}
}

func (s *Server) remove(w http.ResponseWriter, r *http.Request) {
	var req modelRequest
	if err := readJSON(w, r, &req); err != nil {
		apiError(w, 400, err)
		return
	}
	if err := validModel(req.model()); err != nil {
		apiError(w, 400, err)
		return
	}
	names, err := s.backend.Models(r.Context())
	if err != nil {
		apiError(w, 503, errors.New("cannot check running models; stop the backend and use govllm rm locally"))
		return
	}
	for _, name := range names {
		if name == req.model() {
			apiError(w, 409, errors.New("model is in use; stop its vLLM engine before removing it"))
			return
		}
	}
	if err := s.manager.Remove(req.model()); err != nil {
		apiError(w, 404, err)
		return
	}
	writeJSON(w, 200, map[string]string{"status": "success"})
}

type ollamaRequest struct {
	Model    string                     `json:"model"`
	Messages []Message                  `json:"messages"`
	Prompt   string                     `json:"prompt"`
	System   string                     `json:"system"`
	Stream   *bool                      `json:"stream"`
	Options  map[string]json.RawMessage `json:"options"`
	Format   json.RawMessage            `json:"format"`
	Tools    json.RawMessage            `json:"tools"`
	Images   []string                   `json:"images"`
	Raw      bool                       `json:"raw"`
	Template string                     `json:"template"`
	Context  []int                      `json:"context"`
}

func (s *Server) generate(w http.ResponseWriter, r *http.Request, chat bool) {
	started := time.Now()
	var req ollamaRequest
	if err := readJSON(w, r, &req); err != nil {
		apiError(w, 400, err)
		return
	}
	if req.Model == "" {
		apiError(w, 400, errors.New("model is required"))
		return
	}
	if len(req.Tools) > 0 || len(req.Images) > 0 || req.Raw || req.Template != "" || len(req.Context) > 0 {
		apiError(w, 400, errors.New("tools, images, raw prompts, templates, and token contexts are not supported by the Ollama adapter; use /v1 for advanced vLLM features"))
		return
	}
	stream := req.Stream == nil || *req.Stream
	body := map[string]any{"model": req.Model, "stream": stream}
	// Both Ollama routes use vLLM's chat endpoint so instruction models receive
	// their tokenizer's real chat template instead of a hand-built prompt.
	messages := req.Messages
	if !chat {
		messages = []Message{}
		if req.System != "" {
			messages = append(messages, Message{Role: "system", Content: req.System})
		}
		messages = append(messages, Message{Role: "user", Content: req.Prompt})
	}
	if len(messages) == 0 {
		apiError(w, 400, errors.New("messages must not be empty"))
		return
	}
	for _, message := range messages {
		if len(message.Images) > 0 || len(message.ToolCalls) > 0 {
			apiError(w, 400, errors.New("images and tools are not supported by the Ollama adapter; use /v1"))
			return
		}
	}
	body["messages"] = messages
	for from, to := range map[string]string{"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "num_predict": "max_tokens", "stop": "stop", "seed": "seed", "repeat_penalty": "repetition_penalty"} {
		if value, ok := req.Options[from]; ok {
			body[to] = value
		}
	}
	if len(req.Format) > 0 && string(req.Format) != "null" && string(req.Format) != `""` {
		if string(req.Format) == `"json"` {
			body["response_format"] = map[string]any{"type": "json_object"}
		} else {
			var schema map[string]any
			if err := json.Unmarshal(req.Format, &schema); err != nil || schema == nil {
				apiError(w, 400, errors.New("format must be json or a JSON schema object"))
				return
			}
			body["response_format"] = map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "response", "schema": schema}}
		}
	}
	if stream {
		body["stream_options"] = map[string]bool{"include_usage": true}
	}
	resp, err := postJSON(r.Context(), s.backend, "/v1/chat/completions", body)
	if err != nil {
		apiError(w, 502, err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, io.LimitReader(resp.Body, 1<<20))
		return
	}
	response := func(content, thinking string, done bool, reason string, chunk completionChunk) map[string]any {
		value := map[string]any{"model": req.Model, "created_at": time.Now().UTC().Format(time.RFC3339Nano), "done": done}
		if chat {
			value["message"] = map[string]any{"role": "assistant", "content": content}
			if thinking != "" {
				value["message"].(map[string]any)["thinking"] = thinking
			}
		} else {
			value["response"] = content
			if thinking != "" {
				value["thinking"] = thinking
			}
		}
		if done {
			value["done_reason"] = reason
			value["total_duration"] = time.Since(started).Nanoseconds()
			value["prompt_eval_count"] = chunk.Usage.Prompt
			value["eval_count"] = chunk.Usage.Completion
		}
		return value
	}
	if !stream {
		var chunk completionChunk
		if err := json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&chunk); err != nil || len(chunk.Choices) == 0 || len(chunk.Error) > 0 {
			apiError(w, 502, errors.New("invalid completion from vLLM"))
			return
		}
		choice := chunk.Choices[0]
		reason := "stop"
		if choice.FinishReason != nil {
			reason = *choice.FinishReason
		}
		writeJSON(w, 200, response(choice.Message.Content, choice.Message.Thinking, true, reason, chunk))
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	encoder := json.NewEncoder(w)
	var usage completionChunk
	reason := "stop"
	write := func(value any) error {
		if err := encoder.Encode(value); err != nil {
			return err
		}
		return http.NewResponseController(w).Flush()
	}
	err = consumeSSE(resp.Body, func(chunk completionChunk) error {
		if chunk.Usage.Prompt > 0 || chunk.Usage.Completion > 0 {
			usage.Usage = chunk.Usage
		}
		if len(chunk.Choices) == 0 {
			return nil
		}
		choice := chunk.Choices[0]
		if choice.FinishReason != nil {
			reason = *choice.FinishReason
		}
		if choice.Delta.Content == "" && choice.Delta.Thinking == "" {
			return nil
		}
		return write(response(choice.Delta.Content, choice.Delta.Thinking, false, "", chunk))
	})
	if err != nil {
		_ = write(map[string]string{"error": err.Error()})
		return
	}
	_ = write(response("", "", true, reason, usage))
}
