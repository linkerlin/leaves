package objective_test

import (
	"testing"

	"github.com/linkerlin/leaves/v2/objective"
)

func TestRankObjectives(t *testing.T) {
	for _, name := range []string{"rank:pairwise", "rank:ndcg", "rank:listwise"} {
		f, err := objective.ByNameWithClass(name, 0)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if f.Name() != name {
			t.Errorf("got name %q", f.Name())
		}
		if _, ok := objective.IsRanking(f); !ok {
			t.Errorf("%s: not RankFunc", name)
		}
	}
}

func TestRegisterCustomObjective(t *testing.T) {
	objective.Register("custom:obj", func(numClass int) (objective.Func, error) {
		return objective.SquaredError{}, nil
	})
	f, err := objective.ByNameWithClass("custom:obj", 0)
	if err != nil {
		t.Fatal(err)
	}
	if f.Name() != "reg:squarederror" {
		t.Fatalf("got %q", f.Name())
	}
}

// TestBuiltinRegistryComplete 锁定 Phase B：内置 multi/rank 均走 Register，无 switch 回退。
func TestBuiltinRegistryComplete(t *testing.T) {
	names := []string{
		"reg:squarederror", "binary:logistic",
		"multi:softmax", "multi:softprob",
		"rank:pairwise", "rank:ndcg", "rank:listwise",
		"reg:tweedie", "survival:cox", "survival:aft",
	}
	for _, name := range names {
		nc := 0
		if name == "multi:softmax" || name == "multi:softprob" {
			nc = 3
		}
		f, err := objective.ByNameWithClass(name, nc)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if f == nil {
			t.Fatalf("%s: nil", name)
		}
	}
	if _, err := objective.ByNameWithClass("multi:softmax", 1); err == nil {
		t.Fatal("multi:softmax num_class=1 should fail")
	}
	if _, err := objective.ByNameWithClass("no:such", 0); err == nil {
		t.Fatal("unknown should fail")
	}
	// RegisteredNames 应至少含上述键
	reg := map[string]bool{}
	for _, n := range objective.RegisteredNames() {
		reg[n] = true
	}
	for _, name := range names {
		if !reg[name] {
			t.Errorf("RegisteredNames missing %q", name)
		}
	}
}
