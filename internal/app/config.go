package app

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const Version = "0.1.0"

type Config struct {
	Address, ModelsDir, BackendURL, BackendKey, HFEndpoint, HFToken, VLLMBin string
}

func LoadConfig() (Config, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return Config{}, err
	}
	c := Config{
		Address:    env("GOVLLM_ADDR", "127.0.0.1:11434"),
		ModelsDir:  env("GOVLLM_MODELS_DIR", filepath.Join(home, ".govllm", "models")),
		BackendURL: env("GOVLLM_BACKEND_URL", "http://127.0.0.1:8000"),
		BackendKey: os.Getenv("VLLM_API_KEY"),
		HFEndpoint: env("HF_ENDPOINT", "https://huggingface.co"),
		HFToken:    os.Getenv("HF_TOKEN"),
		VLLMBin:    env("GOVLLM_VLLM_BIN", "vllm"),
	}
	return c, nil
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func baseURL(raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimRight(raw, "/"))
	if err != nil || u == nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("expected an HTTP(S) URL without credentials, query, or fragment")
	}
	u.Path = strings.TrimSuffix(strings.TrimRight(u.Path, "/"), "/v1")
	return u, nil
}

type Model struct {
	Name       string    `json:"name"`
	Revision   string    `json:"revision"`
	Digest     string    `json:"digest"`
	Path       string    `json:"path"`
	Size       int64     `json:"size"`
	ModifiedAt time.Time `json:"modified_at"`
}

type Progress struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}
