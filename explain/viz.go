package explain

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/dmitryikh/leaves/tree"
)

// DumpJSON 以 JSON 格式 dump 森林（类 XGBoost dump 结构）。
func DumpJSON(f *tree.ForestIR, featureNames []string) ([]byte, error) {
	if f == nil {
		return []byte("{}"), nil
	}
	model := dumpModelJSON{
		Name:        f.Name,
		NumFeatures: f.NumFeatures,
		NumTrees:    len(f.Trees),
		BaseScore:   f.BaseScore,
		Trees:       make([]dumpTreeJSON, len(f.Trees)),
	}
	for i, t := range f.Trees {
		model.Trees[i] = buildTreeJSON(&t, i, featureNames)
	}
	return json.MarshalIndent(model, "", "  ")
}

// DumpDOT 以 Graphviz DOT 格式 dump 森林（每棵树一个 subgraph）。
func DumpDOT(f *tree.ForestIR, featureNames []string) string {
	if f == nil {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "digraph forest {\n  rankdir=TB;\n  node [shape=box, fontname=Helvetica];\n")
	for i, t := range f.Trees {
		fmt.Fprintf(&b, "  subgraph cluster_%d {\n    label=\"tree %d\";\n", i, i)
		dumpTreeDOT(&b, &t, i, featureNames)
		fmt.Fprintf(&b, "  }\n")
	}
	b.WriteString("}\n")
	return b.String()
}

type dumpModelJSON struct {
	Name        string         `json:"name"`
	NumFeatures int            `json:"num_features"`
	NumTrees    int            `json:"num_trees"`
	BaseScore   float64        `json:"base_score,omitempty"`
	Trees       []dumpTreeJSON `json:"trees"`
}

type dumpTreeJSON struct {
	ID    int           `json:"id"`
	Nodes []dumpNodeJSON `json:"nodes"`
}

type dumpNodeJSON struct {
	ID             int     `json:"id"`
	IsLeaf         bool    `json:"is_leaf"`
	SplitFeature   string  `json:"split_feature,omitempty"`
	SplitCondition float64 `json:"split_condition,omitempty"`
	DefaultLeft    bool    `json:"default_left,omitempty"`
	LeftChild      int     `json:"left_child,omitempty"`
	RightChild     int     `json:"right_child,omitempty"`
	LeafValue      float64 `json:"leaf_value,omitempty"`
	SplitGain      float64 `json:"split_gain,omitempty"`
	Cover          float64 `json:"cover,omitempty"`
}

func buildTreeJSON(t *tree.TreeIR, treeID int, names []string) dumpTreeJSON {
	out := dumpTreeJSON{ID: treeID}
	if t.NumNodes == 0 {
		if len(t.LeafValue) > 0 {
			out.Nodes = append(out.Nodes, dumpNodeJSON{
				ID: 0, IsLeaf: true, LeafValue: t.LeafValue[0],
			})
		}
		return out
	}
	for i := 0; i < t.NumNodes; i++ {
		n := dumpNodeJSON{
			ID:             i,
			IsLeaf:         false,
			SplitFeature:   featureNameOrDefault(names, int(t.SplitFeature[i])),
			SplitCondition: t.SplitThreshold[i],
			LeftChild:      dumpChildRef(t, t.LeftChild[i]),
			RightChild:     dumpChildRef(t, t.RightChild[i]),
		}
		if i < len(t.DefaultLeft) {
			n.DefaultLeft = t.DefaultLeft[i]
		}
		if i < len(t.SplitGain) {
			n.SplitGain = t.SplitGain[i]
		}
		if i < len(t.SumHess) {
			n.Cover = t.SumHess[i]
		}
		out.Nodes = append(out.Nodes, n)
	}
	for li, v := range t.LeafValue {
		out.Nodes = append(out.Nodes, dumpNodeJSON{
			ID: leafDumpID(t, li), IsLeaf: true, LeafValue: v,
		})
	}
	return out
}

func dumpTreeDOT(b *strings.Builder, t *tree.TreeIR, treeID int, names []string) {
	if t.NumNodes == 0 {
		if len(t.LeafValue) > 0 {
			nid := dotNodeID(treeID, 0)
			fmt.Fprintf(b, "    %s [label=\"leaf=%g\"];\n", nid, t.LeafValue[0])
		}
		return
	}
	dumpDOTNode(b, t, treeID, 0, names)
}

func dumpDOTNode(b *strings.Builder, t *tree.TreeIR, treeID int, nodeIdx int32, names []string) {
	if nodeIdx < 0 {
		leafIdx := int(^nodeIdx)
		nid := dotNodeID(treeID, leafDumpID(t, leafIdx))
		val := 0.0
		if leafIdx >= 0 && leafIdx < len(t.LeafValue) {
			val = t.LeafValue[leafIdx]
		}
		fmt.Fprintf(b, "    %s [label=\"leaf=%g\", style=filled, fillcolor=\"#e8f4e8\"];\n", nid, val)
		return
	}
	ni := int(nodeIdx)
	nid := dotNodeID(treeID, ni)
	name := featureNameOrDefault(names, int(t.SplitFeature[ni]))
	cond := "<="
	if ni < len(t.IsCategorical) && t.IsCategorical[ni] {
		cond = "=="
	}
	fmt.Fprintf(b, "    %s [label=\"%s%s%g\"];\n", nid, name, cond, t.SplitThreshold[ni])
	leftID := dotNodeID(treeID, dumpChildRef(t, t.LeftChild[ni]))
	rightID := dotNodeID(treeID, dumpChildRef(t, t.RightChild[ni]))
	fmt.Fprintf(b, "    %s -> %s [label=\"yes\"];\n", nid, leftID)
	fmt.Fprintf(b, "    %s -> %s [label=\"no\"];\n", nid, rightID)
	dumpDOTNode(b, t, treeID, t.LeftChild[ni], names)
	dumpDOTNode(b, t, treeID, t.RightChild[ni], names)
}

func dotNodeID(treeID, nodeID int) string {
	return fmt.Sprintf("t%d_n%d", treeID, nodeID)
}

func dumpChildRef(t *tree.TreeIR, c int32) int {
	if c < 0 {
		return leafDumpID(t, int(^c))
	}
	return int(c)
}

func leafDumpID(t *tree.TreeIR, leafIdx int) int {
	return t.NumNodes + leafIdx
}
