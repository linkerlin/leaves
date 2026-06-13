//go:build born_train

package treebuilder

import "testing"

func TestBornHistAvailable(t *testing.T) {
	if !BornHistAvailable() {
		t.Fatal("expected born_train")
	}
}

func TestScanHistGainsBornMatchesCPU(t *testing.T) {
	histG := []float64{1, 2, 3, 4, 5}
	histH := []float64{1, 1, 1, 1, 1}
	sumG, sumH := 15.0, 5.0
	lambda := 1.0
	sCPU, gCPU := scanHistGainsCPU(histG, histH, sumG, sumH, lambda)
	sBorn, gBorn := scanHistGains(histG, histH, sumG, sumH, lambda, false)
	if sCPU != sBorn || gCPU != gBorn {
		t.Errorf("cpu=(%d,%v) born=(%d,%v)", sCPU, gCPU, sBorn, gBorn)
	}
}
