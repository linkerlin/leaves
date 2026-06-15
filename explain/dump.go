package explain

import (
	"fmt"
	"strings"

	"github.com/linkerlin/leaves/tree"
)

// DumpText 以类 XGBoost 文本格式 dump 森林（简化版）。
func DumpText(f *tree.ForestIR, featureNames []string) string {
	if f == nil {
		return ""
	}
	var b strings.Builder
	for i, t := range f.Trees {
		if i > 0 {
			b.WriteByte('\n')
		}
		fmt.Fprintf(&b, "booster[%d]:\n", i)
		if t.NumNodes == 0 {
			if len(t.LeafValue) > 0 {
				fmt.Fprintf(&b, "0:%g\n", t.LeafValue[0])
			}
			continue
		}
		dumpNode(&b, &t, 0, 0, featureNames)
	}
	return b.String()
}

func dumpNode(b *strings.Builder, t *tree.TreeIR, nodeIdx int32, depth int, names []string) {
	indent := strings.Repeat("\t", depth)
	if nodeIdx < 0 {
		leafIdx := int(^nodeIdx)
		val := 0.0
		if leafIdx >= 0 && leafIdx < len(t.LeafValue) {
			val = t.LeafValue[leafIdx]
		}
		fmt.Fprintf(b, "%s%d:leaf=%g\n", indent, leafDumpID(t, leafIdx), val)
		return
	}
	ni := int(nodeIdx)
	feat := int(t.SplitFeature[ni])
	name := featureNameOrDefault(names, feat)
	cond := "<"
	if ni < len(t.IsCategorical) && t.IsCategorical[ni] {
		cond = "=="
	} else {
		cond = "<="
	}
	dl := 0
	if ni < len(t.DefaultLeft) && t.DefaultLeft[ni] {
		dl = 1
	}
	fmt.Fprintf(b, "%s%d:[%s%s%g] yes=%d,no=%d,missing=%d\n",
		indent, ni, name, cond, t.SplitThreshold[ni],
		dumpChildRef(t, t.LeftChild[ni]),
		dumpChildRef(t, t.RightChild[ni]),
		dl,
	)
	dumpNode(b, t, t.LeftChild[ni], depth+1, names)
	dumpNode(b, t, t.RightChild[ni], depth+1, names)
}
