package leaves_test

import (
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/io"
)

func TestEnsembleReload(t *testing.T) {
	path := filepath.Join("testdata", "xgboost_smoke.json")
	m, err := leaves.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	nFeat := m.NFeatures()
	nEst := m.NEstimators()

	x := make([]float64, nFeat)
	x[1] = 0.7
	before := m.PredictSingle(x, nEst)

	if err := m.Reload(path, &io.LoadOptions{LoadTransformation: false}); err != nil {
		t.Fatal(err)
	}
	if m.NFeatures() != nFeat {
		t.Fatalf("features after reload: %d", m.NFeatures())
	}
	after := m.PredictSingle(x, nEst)
	if before != after {
		t.Fatalf("predictions differ: %v vs %v", before, after)
	}
}
