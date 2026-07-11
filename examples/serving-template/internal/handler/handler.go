package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/example/leaves-serving/internal/modelhost"
)

// API HTTP 处理器集合。
type API struct {
	Host     *modelhost.Host
	MaxBatch int
	// 简易计数（演示用；生产可换 Prometheus）
	PredRequests atomic.Uint64
	PredRows     atomic.Uint64
	PredErrors   atomic.Uint64
	StartedAt    time.Time
}

type errorBody struct {
	Error string `json:"error"`
	Hint  string `json:"hint,omitempty"`
}

type predictRequest struct {
	Features []float64   `json:"features"`
	Rows     [][]float64 `json:"rows"`
	Batch    [][]float64 `json:"batch"`
	NRows    int         `json:"nrows"`
	NCols    int         `json:"ncols"`
}

type predictResponse struct {
	// 单输出单条
	Prediction float64 `json:"prediction,omitempty"`
	// 单输出批 或 多输出扁平
	Predictions []float64 `json:"predictions,omitempty"`
	// 多输出时可选：每行一个向量
	Matrix [][]float64 `json:"matrix,omitempty"`
	NRows  int         `json:"nrows,omitempty"`
	NOut   int         `json:"n_outputs,omitempty"`
}

type metaResponse struct {
	ModelPath string `json:"model_path"`
	NFeatures int    `json:"n_features"`
	NOutputs  int    `json:"n_outputs"`
	Ready     bool   `json:"ready"`
}

type metricsResponse struct {
	UptimeSec    float64 `json:"uptime_sec"`
	PredRequests uint64  `json:"pred_requests"`
	PredRows     uint64  `json:"pred_rows"`
	PredErrors   uint64  `json:"pred_errors"`
}

type reloadRequest struct {
	Path string `json:"path"`
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg, hint string) {
	writeJSON(w, code, errorBody{Error: msg, Hint: hint})
}

// Health liveness：进程在即可。
func (a *API) Health(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

// Ready 模型已加载。
func (a *API) Ready(w http.ResponseWriter, _ *http.Request) {
	if a.Host == nil || !a.Host.Ready() {
		writeErr(w, http.StatusServiceUnavailable, "model not ready", "set LEAVES_MODEL and restart, or POST /admin/reload")
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ready"))
}

// Meta 模型元数据。
func (a *API) Meta(w http.ResponseWriter, _ *http.Request) {
	if a.Host == nil {
		writeErr(w, http.StatusServiceUnavailable, "no host", "")
		return
	}
	nf, no := a.Host.Meta()
	writeJSON(w, http.StatusOK, metaResponse{
		ModelPath: a.Host.Path(),
		NFeatures: nf,
		NOutputs:  no,
		Ready:     a.Host.Ready(),
	})
}

// Metrics 简易计数。
func (a *API) Metrics(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, metricsResponse{
		UptimeSec:    time.Since(a.StartedAt).Seconds(),
		PredRequests: a.PredRequests.Load(),
		PredRows:     a.PredRows.Load(),
		PredErrors:   a.PredErrors.Load(),
	})
}

// Reload 热加载模型（演示用；生产应加鉴权）。
func (a *API) Reload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "POST only", "")
		return
	}
	var req reloadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error(), "JSON {\"path\":\"/models/m.leaves.json\"}")
		return
	}
	if req.Path == "" {
		writeErr(w, http.StatusBadRequest, "path required", "")
		return
	}
	if err := a.Host.Reload(req.Path); err != nil {
		a.PredErrors.Add(1)
		writeErr(w, http.StatusBadRequest, err.Error(), "check path and model format")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "path": req.Path})
}

// Predict 批/单条预测。
func (a *API) Predict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "POST only", "")
		return
	}
	a.PredRequests.Add(1)

	var req predictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		a.PredErrors.Add(1)
		writeErr(w, http.StatusBadRequest, err.Error(), "valid JSON body required")
		return
	}

	rows := req.Rows
	if len(rows) == 0 {
		rows = req.Batch
	}

	nf, nOut := a.Host.Meta()
	if nOut < 1 {
		nOut = 1
	}

	var (
		vals  []float64
		nrows int
		err   error
	)

	switch {
	case len(rows) > 0:
		nrows = len(rows)
		if nrows > a.MaxBatch {
			a.PredErrors.Add(1)
			writeErr(w, http.StatusBadRequest,
				fmt.Sprintf("batch size %d exceeds max %d", nrows, a.MaxBatch),
				"split batch or raise LEAVES_MAX_BATCH")
			return
		}
		vals, err = flattenRows(rows, nf)
	case len(req.Features) > 0 && (req.NRows > 1 || (req.NCols > 0 && len(req.Features) > req.NCols)):
		ncols := req.NCols
		if ncols <= 0 {
			ncols = nf
		}
		nrows = req.NRows
		if nrows <= 0 {
			if ncols > 0 && len(req.Features)%ncols == 0 {
				nrows = len(req.Features) / ncols
			} else {
				a.PredErrors.Add(1)
				writeErr(w, http.StatusBadRequest, "specify nrows/ncols for flat matrix", "")
				return
			}
		}
		if nrows > a.MaxBatch {
			a.PredErrors.Add(1)
			writeErr(w, http.StatusBadRequest, "batch too large", "raise LEAVES_MAX_BATCH")
			return
		}
		need := nrows * ncols
		if len(req.Features) < need {
			a.PredErrors.Add(1)
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("features len %d < %d", len(req.Features), need), "")
			return
		}
		if ncols != nf {
			// 允许调用方写 ncols=n_features；不一致则截断/校验
			if ncols < nf {
				a.PredErrors.Add(1)
				writeErr(w, http.StatusBadRequest, fmt.Sprintf("ncols %d < model n_features %d", ncols, nf), "")
				return
			}
		}
		// 取每行前 nf 列
		vals = make([]float64, 0, nrows*nf)
		for i := 0; i < nrows; i++ {
			off := i * ncols
			vals = append(vals, req.Features[off:off+nf]...)
		}
	case len(req.Features) > 0:
		if len(req.Features) < nf {
			a.PredErrors.Add(1)
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("need %d features, got %d", nf, len(req.Features)), "")
			return
		}
		p := a.Host.PredictSingle(req.Features[:nf])
		a.PredRows.Add(1)
		writeJSON(w, http.StatusOK, predictResponse{Prediction: p, NRows: 1, NOut: nOut})
		return
	default:
		a.PredErrors.Add(1)
		writeErr(w, http.StatusBadRequest, "empty features or rows",
			`{"features":[...]} or {"rows":[[...],[...]]}`)
		return
	}
	if err != nil {
		a.PredErrors.Add(1)
		writeErr(w, http.StatusBadRequest, err.Error(), "")
		return
	}

	out := make([]float64, nrows*nOut)
	if err := a.Host.PredictDense(vals, nrows, nf, out); err != nil {
		a.PredErrors.Add(1)
		writeErr(w, http.StatusInternalServerError, err.Error(), "check feature layout")
		return
	}
	a.PredRows.Add(uint64(nrows))

	resp := predictResponse{NRows: nrows, NOut: nOut}
	if nOut == 1 {
		preds := make([]float64, nrows)
		copy(preds, out)
		if nrows == 1 {
			resp.Prediction = preds[0]
		}
		resp.Predictions = preds
	} else {
		resp.Predictions = out
		mat := make([][]float64, nrows)
		for i := 0; i < nrows; i++ {
			row := make([]float64, nOut)
			copy(row, out[i*nOut:(i+1)*nOut])
			mat[i] = row
		}
		resp.Matrix = mat
	}
	writeJSON(w, http.StatusOK, resp)
}

func flattenRows(rows [][]float64, nf int) ([]float64, error) {
	vals := make([]float64, 0, len(rows)*nf)
	for i, row := range rows {
		if len(row) < nf {
			return nil, fmt.Errorf("row %d: need %d features, got %d", i, nf, len(row))
		}
		vals = append(vals, row[:nf]...)
	}
	return vals, nil
}
