package objective_test

import (
	"testing"

	"github.com/dmitryikh/leaves/objective"
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
