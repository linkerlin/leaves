// 最小 HTTP 推理示例（非官方 serving；embed 于自有服务）。
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/dmitryikh/leaves"
)

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
		var req struct {
			Features []float64 `json:"features"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(req.Features) == 0 {
			http.Error(w, "empty features", http.StatusBadRequest)
			return
		}
		p := m.PredictSingle(req.Features, 0)
		_ = json.NewEncoder(w).Encode(map[string]float64{"prediction": p})
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
