package xgbin

import (
	"bufio"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestReadGBTree(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "xgagaricus.model")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)
	modelHeader, err := ReadModelHeader(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	trueModelHeader := &ModelHeader{}
	trueModelHeader.Param.NumFeatures = 127
	trueModelHeader.NameObj = "binary:logistic"
	trueModelHeader.NameGbm = "gbtree"
	if !reflect.DeepEqual(trueModelHeader, modelHeader) {
		t.Fatalf("unexpected ModelHeader values (got %v)", modelHeader)
	}

	gBTreeModel, err := ReadGBTreeModel(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	trueGBTreeModelParam := GBTreeModelParam{}
	trueGBTreeModelParam.NumTrees = 3
	trueGBTreeModelParam.NumRoots = 1
	trueGBTreeModelParam.NumFeature = 127
	trueGBTreeModelParam.NumOutputGroup = 1
	if !reflect.DeepEqual(trueGBTreeModelParam, gBTreeModel.Param) {
		t.Fatalf("unexpected GBTreeModelParam values (got %v)", gBTreeModel.Param)
	}

	trueTreeParamSlice := [3]TreeParam{}
	trueTreeParamSlice[0].NumRoots = 1
	trueTreeParamSlice[0].NumNodes = 7
	trueTreeParamSlice[0].NumFeature = 127
	trueTreeParamSlice[1].NumRoots = 1
	trueTreeParamSlice[1].NumNodes = 5
	trueTreeParamSlice[1].NumFeature = 127
	trueTreeParamSlice[2].NumRoots = 1
	trueTreeParamSlice[2].NumNodes = 7
	trueTreeParamSlice[2].NumFeature = 127
	if int32(len(gBTreeModel.Trees)) != gBTreeModel.Param.NumTrees {
		t.Fatalf("unexpected len(gBTreeModel.Trees) (got %d", len(gBTreeModel.Trees))
	}
	for i, tree := range gBTreeModel.Trees {
		if !reflect.DeepEqual(trueTreeParamSlice[i], tree.Param) {
			t.Fatalf("unexpected TreeParam values (got %v)", tree.Param)
		}
	}
	// Golden snapshot of tree 0's 7 nodes (xgagaricus.model). Catches float/int
	// parsing regressions in ReadGBTreeModel. Info is the leaf-value/threshold union.
	wantTree0 := []Node{
		{Parent: -1, CLeft: 1, CRight: 2, SIndex: 2147483677, Info: -9.536743e-07},
		{Parent: -2147483648, CLeft: 3, CRight: 4, SIndex: 2147483704, Info: -9.536743e-07},
		{Parent: 0, CLeft: 5, CRight: 6, SIndex: 2147483757, Info: -9.536743e-07},
		{Parent: -2147483647, CLeft: -1, CRight: -1, SIndex: 0, Info: 1.7121772},
		{Parent: 1, CLeft: -1, CRight: -1, SIndex: 0, Info: -1.7004405},
		{Parent: -2147483646, CLeft: -1, CRight: -1, SIndex: 0, Info: -1.9407086},
		{Parent: 2, CLeft: -1, CRight: -1, SIndex: 0, Info: 1.8596492},
	}
	if got := gBTreeModel.Trees[0].Nodes; !reflect.DeepEqual(wantTree0, got) {
		t.Fatalf("tree0 nodes mismatch:\nwant=%+v\ngot =%+v", wantTree0, got)
	}
}
