package train_test

import (
	"testing"

	"github.com/linkerlin/leaves/train"
	"github.com/linkerlin/leaves/treebuilder"
)

func TestResolveTrainTreeMethodWithAccel(t *testing.T) {
	res, gpu := train.ResolveTrainTreeMethodWithAccel(
		treebuilder.MethodAuto, 50000, treebuilder.AccelModeWebGPU, true,
	)
	if res != treebuilder.MethodGPUHist || !gpu {
		t.Fatalf("auto+webgpu 50k: got method=%q useGPU=%v", res, gpu)
	}

	res, gpu = train.ResolveTrainTreeMethodWithAccel(
		treebuilder.MethodHist, 50000, treebuilder.AccelModeWebGPU, true,
	)
	if res != treebuilder.MethodHist || gpu {
		t.Fatalf("explicit hist: got method=%q useGPU=%v", res, gpu)
	}

	res, gpu = train.ResolveTrainTreeMethodWithAccel(
		treebuilder.MethodAuto, 1000, treebuilder.AccelModeCPU, true,
	)
	if res != treebuilder.MethodExact {
		t.Fatalf("auto small: got %q", res)
	}
}
