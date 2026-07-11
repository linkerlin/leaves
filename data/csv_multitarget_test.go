package data_test

import (
	"strings"
	"testing"

	"github.com/linkerlin/leaves/data"
)

func TestCSVMultiTargetTrailing(t *testing.T) {
	csv := "f0,f1,y0,y1\n0.1,0.2,0.3,0.4\n1,2,3,4\n"
	dm, err := data.FromCSVReader(strings.NewReader(csv), data.CSVOptions{
		HasHeader:          true,
		NumTrailingTargets: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if dm.NumTarget() != 2 {
		t.Fatalf("NumTarget=%d", dm.NumTarget())
	}
	if dm.NumCol() != 2 || dm.NumRow() != 2 {
		t.Fatalf("shape %d x %d", dm.NumRow(), dm.NumCol())
	}
	tg := dm.Targets()
	if len(tg) != 4 || tg[0] != 0.3 || tg[1] != 0.4 || tg[2] != 3 || tg[3] != 4 {
		t.Fatalf("targets=%v", tg)
	}
	// Labels 为第 0 目标
	if dm.Labels()[0] != 0.3 || dm.Labels()[1] != 3 {
		t.Fatalf("labels0=%v", dm.Labels())
	}
}
