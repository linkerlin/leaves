package data

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFromCSVReader(t *testing.T) {
	const csv = "a,b,label\n1,2,0\n3,4,1\n"
	dm, err := FromCSVReader(strings.NewReader(csv), CSVOptions{HasHeader: true, HasLabelColumn: true, LabelCol: 2})
	if err != nil {
		t.Fatal(err)
	}
	if dm.NumRow() != 2 || dm.NumCol() != 2 {
		t.Fatalf("shape %dx%d", dm.NumRow(), dm.NumCol())
	}
	if got := dm.Labels(); len(got) != 2 || got[0] != 0 || got[1] != 1 {
		t.Fatalf("labels %v", got)
	}
}

func TestFromCSVFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")
	content := "feat0,feat1\n1,0\n2,1\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	dm, err := FromCSV(path, CSVOptions{HasHeader: true})
	if err != nil {
		t.Fatal(err)
	}
	if dm.NumRow() != 2 || dm.NumCol() != 2 {
		t.Fatalf("shape %dx%d", dm.NumRow(), dm.NumCol())
	}
}
