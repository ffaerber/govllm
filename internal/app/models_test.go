package app

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

const fixtureSHA = "0123456789abcdef0123456789abcdef01234567"

type hubFixture struct {
	server    *httptest.Server
	downloads atomic.Int32
	corrupt   atomic.Bool
	manifest  func() hubRepo
}

func newHub(t *testing.T) *hubFixture {
	t.Helper()
	h := &hubFixture{}
	files := map[string]string{"config.json": `{"model_type":"test"}`, "model.safetensors": "fixture weights", "tokenizer.json": `{"version":"1.0"}`}
	h.manifest = func() hubRepo {
		repo := hubRepo{SHA: fixtureSHA}
		for name, data := range files {
			blob := sha1.Sum([]byte(fmt.Sprintf("blob %d\x00%s", len(data), data)))
			repo.Files = append(repo.Files, hubFile{Name: name, Size: int64(len(data)), BlobID: hex.EncodeToString(blob[:])})
		}
		// LFS weights have a SHA-256 content digest, not a Git blob hash.
		for i := range repo.Files {
			if repo.Files[i].Name == "model.safetensors" {
				data, _ := json.Marshal(map[string]any{"rfilename": "model.safetensors", "size": len(files["model.safetensors"]), "lfs": map[string]any{"sha256": fmt.Sprintf("%x", sha256.Sum256([]byte(files["model.safetensors"]))), "size": len(files["model.safetensors"])}})
				_ = json.Unmarshal(data, &repo.Files[i])
			}
		}
		return repo
	}
	h.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer fixture-token" {
			http.Error(w, "unauthorized", 401)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/models/test/tiny/revision/") {
			if r.URL.Query().Get("blobs") != "true" {
				t.Error("missing blob metadata request")
			}
			_ = json.NewEncoder(w).Encode(h.manifest())
			return
		}
		prefix := "/test/tiny/resolve/" + fixtureSHA + "/"
		if !strings.HasPrefix(r.URL.Path, prefix) {
			http.NotFound(w, r)
			return
		}
		h.downloads.Add(1)
		name := strings.TrimPrefix(r.URL.Path, prefix)
		if h.corrupt.Load() {
			_, _ = io.WriteString(w, "corrupt")
			return
		}
		data, ok := files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = io.WriteString(w, data)
	}))
	t.Cleanup(h.server.Close)
	return h
}

func managerFor(t *testing.T, h *hubFixture) *Manager {
	t.Helper()
	m, err := NewManager(Config{ModelsDir: t.TempDir(), HFEndpoint: h.server.URL, HFToken: "fixture-token"})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestModelLifecycle(t *testing.T) {
	h := newHub(t)
	m := managerFor(t, h)
	var statuses []string
	model, err := m.Pull(context.Background(), "test/tiny", "main", func(p Progress) { statuses = append(statuses, p.Status) })
	if err != nil {
		t.Fatal(err)
	}
	if model.Digest != fixtureSHA || model.Size <= 0 || statuses[len(statuses)-1] != "success" {
		t.Fatalf("bad model/progress: %+v %v", model, statuses)
	}
	if data, err := os.ReadFile(filepath.Join(model.Path, "model.safetensors")); err != nil || string(data) != "fixture weights" {
		t.Fatalf("weights: %q %v", data, err)
	}
	listed, err := m.List()
	if err != nil || len(listed) != 1 || listed[0].Name != "test/tiny" {
		t.Fatalf("list: %v %v", listed, err)
	}
	if _, err := m.Pull(context.Background(), "test/tiny", "main", nil); err != nil {
		t.Fatal(err)
	}
	if h.downloads.Load() != 3 {
		t.Fatalf("cached pull downloaded %d files", h.downloads.Load())
	}
	if err := m.Remove("test/tiny"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(model.Path); !os.IsNotExist(err) {
		t.Fatalf("snapshot still exists: %v", err)
	}
	listed, err = m.List()
	if err != nil || len(listed) != 0 {
		t.Fatalf("after delete: %v %v", listed, err)
	}
}

func TestPullIntegrityFailureNeverPublishes(t *testing.T) {
	h := newHub(t)
	h.corrupt.Store(true)
	m := managerFor(t, h)
	if _, err := m.Pull(context.Background(), "test/tiny", "main", nil); err == nil {
		t.Fatal("corrupted download accepted")
	}
	if _, err := os.Stat(m.manifest("test/tiny")); !os.IsNotExist(err) {
		t.Fatalf("partial manifest published: %v", err)
	}
	matches, _ := filepath.Glob(filepath.Join(m.dir, ".pull-*"))
	if len(matches) != 0 {
		t.Fatalf("staging directories leaked: %v", matches)
	}
	h.corrupt.Store(false)
	if _, err := m.Pull(context.Background(), "test/tiny", "main", nil); err != nil {
		t.Fatalf("retry after failure: %v", err)
	}
}

func TestPullCanceledNeverPublishes(t *testing.T) {
	h := newHub(t)
	m := managerFor(t, h)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Pull(ctx, "test/tiny", "main", nil); err == nil {
		t.Fatal("canceled pull succeeded")
	}
	if h.downloads.Load() != 0 {
		t.Fatal("canceled pull downloaded files")
	}
	if _, err := os.Stat(filepath.Join(m.dir, ".mutation-lock")); !os.IsNotExist(err) {
		t.Fatal("lock leaked")
	}
}

func TestModelMutationsRespectCrossProcessLock(t *testing.T) {
	h := newHub(t)
	m := managerFor(t, h)
	if _, err := m.Pull(context.Background(), "test/tiny", "main", nil); err != nil {
		t.Fatal(err)
	}
	unlock, err := m.lockMutation()
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	if err := m.Remove("test/tiny"); err == nil {
		t.Fatal("delete ignored mutation lock")
	}
	if _, err := m.Pull(context.Background(), "test/tiny", "main", nil); err == nil {
		t.Fatal("pull ignored mutation lock")
	}
}

func TestUnsafeManifestsAndUnsupportedModels(t *testing.T) {
	for _, name := range []string{"../etc/passwd", "test/../tiny", "/tmp/model", "test/tiny:Q4_K_M", "test/tiny/extra", "test"} {
		t.Run(name, func(t *testing.T) {
			if err := validModel(name); err == nil {
				t.Fatalf("accepted %q", name)
			}
		})
	}
	for _, name := range []string{"../outside", "/outside", "a/../../outside", "a\\outside", "./config.json"} {
		t.Run("path_"+name, func(t *testing.T) {
			_, err := selectFiles([]hubFile{{Name: "model.safetensors"}, {Name: "config.json"}, {Name: name}})
			if err == nil {
				t.Fatalf("accepted unsafe path %q", name)
			}
		})
	}
	if _, err := selectFiles([]hubFile{{Name: "weights.gguf"}, {Name: "config.json"}}); err == nil {
		t.Fatal("GGUF silently accepted")
	}
	files, err := selectFiles([]hubFile{{Name: "config.json"}, {Name: "model.safetensors"}, {Name: "pytorch_model.bin"}, {Name: "model.py"}, {Name: "onnx/weights.bin"}})
	if err != nil || len(files) != 2 {
		t.Fatalf("file selection: %v %v", files, err)
	}
}

func TestDownloadDoesNotForwardTokenAcrossRedirects(t *testing.T) {
	var gotToken string
	dest := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotToken = r.Header.Get("Authorization")
		fmt.Fprint(w, "ok")
	}))
	defer dest.Close()
	src := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, dest.URL, 302) }))
	defer src.Close()
	m, err := NewManager(Config{ModelsDir: t.TempDir(), HFEndpoint: src.URL, HFToken: "secret-fixture"})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := m.get(context.Background(), src.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if gotToken != "" {
		t.Fatal("token leaked to redirect destination")
	}
}

func TestInvalidManifestDoesNotDeleteOutsideCache(t *testing.T) {
	h := newHub(t)
	m := managerFor(t, h)
	if err := os.MkdirAll(filepath.Dir(m.manifest("test/tiny")), 0700); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	data, _ := json.Marshal(Model{Name: "test/tiny", Digest: fixtureSHA, Path: outside})
	if err := os.WriteFile(m.manifest("test/tiny"), data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := m.Remove("test/tiny"); err == nil {
		t.Fatal("accepted tampered manifest")
	}
	if _, err := os.Stat(outside); err != nil {
		t.Fatal("removed outside directory")
	}
}
