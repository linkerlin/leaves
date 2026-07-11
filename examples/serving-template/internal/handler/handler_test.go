package handler_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/example/leaves-serving/internal/handler"
	"github.com/example/leaves-serving/internal/modelhost"
)

func TestPredictBatch(t *testing.T) {
	modelPath := filepath.Join("..", "..", "..", "testdata", "xgboost_smoke.json")
	host, err := modelhost.Load(modelPath)
	if err != nil {
		t.Skip(err)
	}
	defer host.Close()
	nf, _ := host.Meta()
	api := &handler.API{Host: host, MaxBatch: 100, StartedAt: time.Now()}

	row := make([]float64, nf)
	body, _ := json.Marshal(map[string]any{"rows": [][]float64{row, row}})
	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	api.Predict(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	preds, ok := resp["predictions"].([]any)
	if !ok || len(preds) != 2 {
		t.Fatalf("predictions: %v", resp)
	}
}

func TestHealthReady(t *testing.T) {
	modelPath := filepath.Join("..", "..", "..", "testdata", "xgboost_smoke.json")
	host, err := modelhost.Load(modelPath)
	if err != nil {
		t.Skip(err)
	}
	defer host.Close()
	api := &handler.API{Host: host, MaxBatch: 10, StartedAt: time.Now()}

	rr := httptest.NewRecorder()
	api.Health(rr, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rr.Code != 200 || rr.Body.String() != "ok" {
		t.Fatalf("health: %d %s", rr.Code, rr.Body.String())
	}
	rr2 := httptest.NewRecorder()
	api.Ready(rr2, httptest.NewRequest(http.MethodGet, "/ready", nil))
	if rr2.Code != 200 {
		t.Fatalf("ready: %d", rr2.Code)
	}
}

func TestMaxBatch(t *testing.T) {
	modelPath := filepath.Join("..", "..", "..", "testdata", "xgboost_smoke.json")
	host, err := modelhost.Load(modelPath)
	if err != nil {
		t.Skip(err)
	}
	defer host.Close()
	nf, _ := host.Meta()
	api := &handler.API{Host: host, MaxBatch: 1, StartedAt: time.Now()}
	row := make([]float64, nf)
	body, _ := json.Marshal(map[string]any{"rows": [][]float64{row, row}})
	rr := httptest.NewRecorder()
	api.Predict(rr, httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(body)))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("want 400, got %d %s", rr.Code, rr.Body.String())
	}
}
