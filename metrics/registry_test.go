package metrics_test

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/metrics"
)

func TestResolveMetrics(t *testing.T) {
	cases := []struct {
		name    string
		opt     metrics.Options
		want    string
		wantErr bool
	}{
		{"rmse", metrics.Options{}, "rmse", false},
		{"MAPE", metrics.Options{}, "mape", false},
		{"ndcg@3", metrics.Options{Groups: []int{4}}, "ndcg", false},
		{"mlogloss", metrics.Options{NumClass: 3}, "mlogloss", false},
		{"mlogloss", metrics.Options{NumClass: 1}, "", true},
		{"unknown", metrics.Options{}, "", true},
	}
	for _, tc := range cases {
		m, err := metrics.Resolve(tc.name, tc.opt)
		if tc.wantErr {
			if err == nil {
				t.Errorf("%q: expected error", tc.name)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%q: %v", tc.name, err)
		}
		if m.Name() != tc.want {
			t.Errorf("%q: got name %q want %q", tc.name, m.Name(), tc.want)
		}
	}
}

func TestNormalizeNameAliases(t *testing.T) {
	if metrics.NormalizeName("binary:logistic") != "logloss" {
		t.Fatal("binary:logistic alias")
	}
	if metrics.NormalizeName("reg:squarederror") != "rmse" {
		t.Fatal("reg:squarederror alias")
	}
}

// TestResolveViaRegistryOnly 锁定 Phase B：mlogloss/merror/ndcg@ 均走 Register。
func TestResolveViaRegistryOnly(t *testing.T) {
	m, err := metrics.Resolve("merror", metrics.Options{NumClass: 4})
	if err != nil || m.Name() != "merror" {
		t.Fatalf("merror: %v %#v", err, m)
	}
	// map@3 与 ndcg@3 前缀解析
	m, err = metrics.Resolve("map@3", metrics.Options{Groups: []int{3, 3}})
	if err != nil || m.Name() != "map" {
		t.Fatalf("map@3: %v %#v", err, m)
	}
	// 自定义指标
	metrics.Register("custom_metric", func(o metrics.Options) (metrics.Metric, error) {
		return metrics.RMSE{}, nil
	})
	m, err = metrics.Resolve("custom_metric", metrics.Options{})
	if err != nil || m.Name() != "rmse" {
		t.Fatalf("custom: %v", err)
	}
	reg := map[string]bool{}
	for _, n := range metrics.RegisteredNames() {
		reg[n] = true
	}
	for _, need := range []string{"rmse", "mlogloss", "merror", "ndcg", "map"} {
		if !reg[need] {
			t.Errorf("RegisteredNames missing %q", need)
		}
	}
}

func TestXGBoostTrendRMSELogLoss(t *testing.T) {
	// 与 XGBoost 同公式；训练 float 序差异允许 ±1e-9（单测级）。
	yTrue := []float64{1, 2, 3}
	yPred := []float64{1.1, 1.9, 3.2}
	rmse, err := metrics.RMSE{}.Evaluate(yTrue, yPred)
	if err != nil {
		t.Fatal(err)
	}
	wantRMSE := math.Sqrt((0.01 + 0.01 + 0.04) / 3)
	if math.Abs(rmse-wantRMSE) > 1e-9 {
		t.Errorf("rmse %f want %f", rmse, wantRMSE)
	}
	ll, err := metrics.LogLoss{}.Evaluate([]float64{1, 0}, []float64{0.9, 0.1})
	if err != nil {
		t.Fatal(err)
	}
	const eps = 1e-15
	wantLL := -(math.Log(0.9) + math.Log(0.9)) / 2
	if math.Abs(ll-wantLL) > 1e-6 {
		t.Errorf("logloss %f want %f", ll, wantLL)
	}
	_ = eps
}

// 容忍度说明（文档化）：leaves 与 XGBoost Python 在 MAPE（跳过 y=0）、排序 metric（group 切分）
// 上语义对齐；训练曲线趋势一致即可，不要求 bit-exact（分箱/浮点序不同）。
