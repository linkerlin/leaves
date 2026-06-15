// 最小 HTTP 推理示例（非官方 serving；embed 于自有服务）。
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/dmitryikh/leaves"
	"github.com/dmitryikh/leaves/model"
)

type predictRequest struct {
	Features []float64   `json:"features"` // 单条或扁平矩阵
	Rows     [][]float64 `json:"rows"`     // 批：每行一条样本
	Batch    [][]float64 `json:"batch"`    // rows 别名
	NRows    int         `json:"nrows"`    // 与 features 联用：扁平矩阵行数
	NCols    int         `json:"ncols"`    // 与 features 联用：列数
}

type predictResponse struct {
	Prediction  float64   `json:"prediction,omitempty"`
	Predictions []float64 `json:"predictions,omitempty"`
	NRows       int       `json:"nrows,omitempty"`
}

func main() {
	path := os.Getenv("LEAVES_MODEL")
	if path == "" {
		path = "../../testdata/xgboost_smoke.json"
	}
	m, err := leaves.LoadFromFile(path, leaves.DefaultLoadOptions())
	if err != nil {
		log.Fatal(err)
	}

	http.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req predictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		rows := req.Rows
		if len(rows) == 0 {
			rows = req.Batch
		}
		w.Header().Set("Content-Type", "application/json")
		if len(rows) > 0 {
			resp, err := predictBatch(m, rows)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			_ = json.NewEncoder(w).Encode(resp)
			return
		}
		if len(req.Features) == 0 {
			http.Error(w, "empty features or rows", http.StatusBadRequest)
			return
		}
		if req.NRows > 1 || (req.NCols > 0 && len(req.Features) > req.NCols) {
			resp, err := predictFlat(m, req.Features, req.NRows, req.NCols)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			_ = json.NewEncoder(w).Encode(resp)
			return
		}
		p := m.PredictSingle(req.Features, 0)
		_ = json.NewEncoder(w).Encode(predictResponse{Prediction: p})
	})

	http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	addr := os.Getenv("LEAVES_HTTP_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	log.Printf("leaves http demo on %s model=%s", addr, path)
	log.Fatal(http.ListenAndServe(addr, nil))
}

func predictBatch(m *model.Ensemble, rows [][]float64) (predictResponse, error) {
	if len(rows) == 0 {
		return predictResponse{}, fmt.Errorf("empty batch")
	}
	ncols := m.NFeatures()
	nrows := len(rows)
	vals := make([]float64, 0, nrows*ncols)
	for i, row := range rows {
		if len(row) < ncols {
			return predictResponse{}, fmt.Errorf("row %d: need %d features, got %d", i, ncols, len(row))
		}
		vals = append(vals, row[:ncols]...)
	}
	out := make([]float64, nrows*m.NOutputGroups())
	if err := m.PredictDense(vals, nrows, ncols, out, 0, 0); err != nil {
		return predictResponse{}, err
	}
	if m.NOutputGroups() == 1 {
		preds := make([]float64, nrows)
		copy(preds, out)
		return predictResponse{Predictions: preds, NRows: nrows}, nil
	}
	return predictResponse{Predictions: out, NRows: nrows}, nil
}

func predictFlat(m *model.Ensemble, flat []float64, nrows, ncols int) (predictResponse, error) {
	if ncols <= 0 {
		ncols = m.NFeatures()
	}
	if nrows <= 0 {
		if ncols > 0 && len(flat)%ncols == 0 {
			nrows = len(flat) / ncols
		} else {
			return predictResponse{}, fmt.Errorf("specify nrows/ncols for flat matrix")
		}
	}
	if len(flat) < nrows*ncols {
		return predictResponse{}, fmt.Errorf("features len %d < %d*%d", len(flat), nrows, ncols)
	}
	out := make([]float64, nrows*m.NOutputGroups())
	if err := m.PredictDense(flat[:nrows*ncols], nrows, ncols, out, 0, 0); err != nil {
		return predictResponse{}, err
	}
	if nrows == 1 && m.NOutputGroups() == 1 {
		return predictResponse{Prediction: out[0]}, nil
	}
	preds := out
	if m.NOutputGroups() == 1 {
		preds = make([]float64, nrows)
		copy(preds, out)
	}
	return predictResponse{Predictions: preds, NRows: nrows}, nil
}
