package app

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	ErrNotFound = errors.New("model not found; use govllm pull first")
	modelID     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)
	commitID    = regexp.MustCompile(`^[a-f0-9]{40}$`)
)

type Manager struct {
	dir, endpoint, token string
	client               *http.Client
	mu                   sync.Mutex
}

func NewManager(c Config) (*Manager, error) {
	u, err := baseURL(c.HFEndpoint)
	if err != nil {
		return nil, fmt.Errorf("HF_ENDPOINT: %w", err)
	}
	dir, err := filepath.Abs(c.ModelsDir)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		ResponseHeaderTimeout: 2 * time.Minute,
		TLSHandshakeTimeout:   15 * time.Second,
	}, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 {
			return errors.New("too many download redirects")
		}
		if req.URL.Scheme != "https" && via[0].URL.Scheme == "https" {
			return errors.New("refusing insecure download redirect")
		}
		if req.URL.Host != via[0].URL.Host {
			req.Header.Del("Authorization")
		}
		return nil
	}}
	return &Manager{dir: dir, endpoint: u.String(), token: c.HFToken, client: client}, nil
}

func validModel(name string) error {
	if !modelID.MatchString(name) || strings.Contains(name, "..") || strings.Contains(name, "--") {
		return fmt.Errorf("invalid model %q: use a Hugging Face owner/model ID (GGUF quantization suffixes are not supported)", name)
	}
	return nil
}

func key(name string) string { sum := sha256.Sum256([]byte(name)); return hex.EncodeToString(sum[:]) }
func (m *Manager) manifest(name string) string {
	return filepath.Join(m.dir, "manifests", key(name)+".json")
}

func (m *Manager) Show(name string) (Model, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.show(name)
}

func (m *Manager) show(name string) (Model, error) {
	if err := validModel(name); err != nil {
		return Model{}, err
	}
	b, err := os.ReadFile(m.manifest(name))
	if errors.Is(err, os.ErrNotExist) {
		return Model{}, ErrNotFound
	}
	if err != nil {
		return Model{}, err
	}
	var model Model
	if err := json.Unmarshal(b, &model); err != nil {
		return model, err
	}
	expected := filepath.Join(m.dir, "snapshots", key(name), model.Digest)
	if model.Name != name || !commitID.MatchString(model.Digest) || model.Path != expected {
		return Model{}, errors.New("invalid local model manifest")
	}
	if info, err := os.Stat(model.Path); err != nil || !info.IsDir() {
		return Model{}, errors.New("model snapshot is missing; pull it again")
	}
	return model, nil
}

func (m *Manager) List() ([]Model, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	entries, err := os.ReadDir(filepath.Join(m.dir, "manifests"))
	if errors.Is(err, os.ErrNotExist) {
		return []Model{}, nil
	}
	if err != nil {
		return nil, err
	}
	models := []Model{}
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(m.dir, "manifests", e.Name()))
		if err != nil {
			return nil, err
		}
		var model Model
		if err := json.Unmarshal(b, &model); err != nil {
			return nil, err
		}
		model, err = m.show(model.Name)
		if err != nil {
			return nil, err
		}
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	return models, nil
}

func (m *Manager) Remove(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	unlock, err := m.lockMutation()
	if err != nil {
		return err
	}
	defer unlock()
	if _, err := m.show(name); err != nil {
		return err
	}
	if err := os.Remove(m.manifest(name)); err != nil {
		return err
	}
	return os.RemoveAll(filepath.Join(m.dir, "snapshots", key(name)))
}

func (m *Manager) lockMutation() (func(), error) {
	if err := os.MkdirAll(m.dir, 0700); err != nil {
		return nil, err
	}
	lock := filepath.Join(m.dir, ".mutation-lock")
	f, err := os.OpenFile(lock, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if errors.Is(err, os.ErrExist) {
		return nil, errors.New("another model operation is in progress; if it crashed, remove the .mutation-lock file after checking no model operation is running")
	}
	if err != nil {
		return nil, err
	}
	f.Close()
	return func() { _ = os.Remove(lock) }, nil
}

type hubFile struct {
	Name   string `json:"rfilename"`
	Size   int64  `json:"size"`
	BlobID string `json:"blobId"`
	LFS    *struct {
		SHA  string `json:"sha256"`
		OID  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs"`
}
type hubRepo struct {
	SHA   string    `json:"sha"`
	Files []hubFile `json:"siblings"`
}

// Pull publishes a manifest only after the complete pinned snapshot is verified.
// A filesystem lock coordinates separate CLI and HTTP server processes.
func (m *Manager) Pull(ctx context.Context, name, revision string, progress func(Progress)) (Model, error) {
	if err := validModel(name); err != nil {
		return Model{}, err
	}
	if revision == "" {
		revision = "main"
	}
	if progress == nil {
		progress = func(Progress) {}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	unlock, err := m.lockMutation()
	if err != nil {
		return Model{}, err
	}
	defer unlock()
	progress(Progress{Status: "fetching manifest"})
	api := m.endpoint + "/api/models/" + name + "/revision/" + url.PathEscape(revision) + "?blobs=true"
	resp, err := m.get(ctx, api)
	if err != nil {
		return Model{}, err
	}
	var repo hubRepo
	err = json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&repo)
	resp.Body.Close()
	if err != nil {
		return Model{}, fmt.Errorf("Hugging Face metadata: %w", err)
	}
	if !commitID.MatchString(repo.SHA) {
		return Model{}, errors.New("Hugging Face did not return a valid commit SHA")
	}
	if cached, err := m.show(name); err == nil && cached.Digest == repo.SHA {
		progress(Progress{Status: "success"})
		return cached, nil
	}
	files, err := selectFiles(repo.Files)
	if err != nil {
		return Model{}, err
	}
	stage, err := os.MkdirTemp(m.dir, ".pull-")
	if err != nil {
		return Model{}, err
	}
	defer os.RemoveAll(stage)
	var size int64
	for _, file := range files {
		if err := ctx.Err(); err != nil {
			return Model{}, err
		}
		download := m.endpoint + "/" + name + "/resolve/" + repo.SHA + "/" + escapePath(file.Name)
		progress(Progress{Status: "downloading " + file.Name, Total: file.Size})
		n, err := m.download(ctx, download, filepath.Join(stage, filepath.FromSlash(file.Name)), file)
		if err != nil {
			return Model{}, fmt.Errorf("download %s: %w", file.Name, err)
		}
		size += n
		progress(Progress{Status: "downloaded " + file.Name, Total: n, Completed: n})
	}
	dest := filepath.Join(m.dir, "snapshots", key(name), repo.SHA)
	if err := os.MkdirAll(filepath.Dir(dest), 0700); err != nil {
		return Model{}, err
	}
	// Another revision can be retained for a running backend; rm removes all revisions.
	if _, err := os.Stat(dest); errors.Is(err, os.ErrNotExist) {
		if err := os.Rename(stage, dest); err != nil {
			return Model{}, err
		}
	} else if err != nil {
		return Model{}, err
	}
	model := Model{Name: name, Revision: revision, Digest: repo.SHA, Path: dest, Size: size, ModifiedAt: time.Now().UTC()}
	b, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return Model{}, err
	}
	if err := os.MkdirAll(filepath.Dir(m.manifest(name)), 0700); err != nil {
		return Model{}, err
	}
	tmp, err := os.CreateTemp(filepath.Dir(m.manifest(name)), ".manifest-")
	if err != nil {
		return Model{}, err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return Model{}, err
	}
	if err := tmp.Close(); err != nil {
		return Model{}, err
	}
	if err := os.Rename(tmp.Name(), m.manifest(name)); err != nil {
		return Model{}, err
	}
	progress(Progress{Status: "success"})
	return model, nil
}

func selectFiles(files []hubFile) ([]hubFile, error) {
	var safe, pytorch bool
	for _, f := range files {
		if path.Dir(f.Name) == "." {
			safe = safe || strings.HasSuffix(f.Name, ".safetensors")
			pytorch = pytorch || (strings.HasPrefix(f.Name, "pytorch_model") && strings.HasSuffix(f.Name, ".bin"))
		}
	}
	if !safe && !pytorch {
		return nil, errors.New("no root-level safetensors or PyTorch weights found; this version supports Transformers models, not GGUF/adapters")
	}
	selected := []hubFile{}
	seen := map[string]bool{}
	var config bool
	for _, f := range files {
		if f.Name == "" || strings.Contains(f.Name, "\\") || strings.HasPrefix(f.Name, "/") || path.Clean(f.Name) != f.Name || f.Name == ".." || strings.HasPrefix(f.Name, "../") || strings.ContainsRune(f.Name, 0) {
			return nil, errors.New("unsafe path in Hugging Face manifest")
		}
		if path.Dir(f.Name) != "." {
			continue
		}
		ext := path.Ext(f.Name)
		include := ext == ".json" || ext == ".txt" || ext == ".model" || ext == ".jinja" || ext == ".tiktoken" || (safe && ext == ".safetensors") || (!safe && ext == ".bin" && strings.HasPrefix(f.Name, "pytorch_model"))
		if !include {
			continue
		}
		if seen[f.Name] {
			return nil, errors.New("duplicate file in Hugging Face manifest")
		}
		seen[f.Name] = true
		config = config || f.Name == "config.json"
		if f.LFS != nil {
			f.Size = f.LFS.Size
		}
		if f.Size < 0 {
			return nil, errors.New("invalid file size")
		}
		selected = append(selected, f)
	}
	if !config {
		return nil, errors.New("model repository has no config.json")
	}
	return selected, nil
}

func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}

func (m *Manager) get(ctx context.Context, raw string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, raw, nil)
	if err != nil {
		return nil, err
	}
	if m.token != "" {
		req.Header.Set("Authorization", "Bearer "+m.token)
	}
	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Hugging Face request failed (check network access)")
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("Hugging Face HTTP %d (gated/private models require HF_TOKEN and license access)", resp.StatusCode)
	}
	return resp, nil
}

func (m *Manager) download(ctx context.Context, raw, dest string, info hubFile) (int64, error) {
	resp, err := m.get(ctx, raw)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if err := os.MkdirAll(filepath.Dir(dest), 0700); err != nil {
		return 0, err
	}
	f, err := os.OpenFile(dest, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	var h hash.Hash
	var expected string
	if info.LFS != nil {
		h = sha256.New()
		expected = info.LFS.SHA
		if expected == "" {
			expected = strings.TrimPrefix(info.LFS.OID, "sha256:")
		}
	} else {
		h = sha1.New()
		expected = info.BlobID
		fmt.Fprintf(h, "blob %d\x00", info.Size)
	}
	if expected == "" {
		return 0, errors.New("manifest has no file checksum")
	}
	n, err := io.Copy(io.MultiWriter(f, h), io.LimitReader(resp.Body, info.Size+1))
	if err != nil {
		return n, err
	}
	if n != info.Size {
		return n, fmt.Errorf("size mismatch: expected %d, received %d", info.Size, n)
	}
	if hex.EncodeToString(h.Sum(nil)) != expected {
		return n, errors.New("checksum mismatch")
	}
	if err := f.Close(); err != nil {
		return n, err
	}
	return n, nil
}
