package data

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSniffLIBSVM(t *testing.T) {
	path := filepath.Join("..", "testdata", "csrmat.libsvm")
	sniff, err := SniffFileFormat(path)
	if err != nil {
		t.Fatal(err)
	}
	if sniff.Format != FormatLIBSVM {
		t.Fatalf("format=%v want LIBSVM", sniff.Format)
	}
	if !sniff.LIBSVM.HasLabel {
		t.Fatal("expected HasLabel=true for libsvm sniff")
	}
}

func TestFromFileAutoLIBSVM(t *testing.T) {
	path := filepath.Join("..", "testdata", "csrmat.libsvm")
	m, err := FromFileAuto(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.NumRow() == 0 {
		t.Fatal("empty matrix")
	}
}

func TestSniffRankingTSV(t *testing.T) {
	path := filepath.Join("..", "testdata", "rank_smoke_train.tsv")
	sniff, err := SniffFileFormat(path)
	if err != nil {
		t.Fatal(err)
	}
	if sniff.Format != FormatRanking {
		t.Fatalf("format=%v want Ranking", sniff.Format)
	}
}

func TestFromFileAutoRanking(t *testing.T) {
	path := filepath.Join("..", "testdata", "rank_smoke_train.tsv")
	m, err := FromFileAuto(path)
	if err != nil {
		t.Fatal(err)
	}
	dg, ok := m.(*DenseWithGroups)
	if !ok {
		t.Fatalf("got %T want *DenseWithGroups", m)
	}
	if dg.NumRow() == 0 || len(dg.Groups()) == 0 {
		t.Fatal("empty ranking matrix")
	}
}

func TestSniffTSVLabelLast(t *testing.T) {
	path := filepath.Join("..", "testdata", "breast_cancer_train.tsv")
	sniff, err := SniffFileFormat(path)
	if err != nil {
		t.Fatal(err)
	}
	if sniff.Format != FormatTSVLabelLast {
		t.Fatalf("format=%v want TSVLabelLast", sniff.Format)
	}
}

func TestFromFileAutoDenseTSV(t *testing.T) {
	path := filepath.Join("..", "testdata", "breast_cancer_train.tsv")
	m, err := FromFileAuto(path)
	if err != nil {
		t.Fatal(err)
	}
	d, ok := m.(*Dense)
	if !ok {
		t.Fatalf("got %T want *Dense", m)
	}
	if d.NumRow() < 10 || d.NumCol() < 2 {
		t.Fatalf("unexpected shape %dx%d", d.NumRow(), d.NumCol())
	}
	if len(d.Labels()) != d.NumRow() {
		t.Fatalf("labels %d != rows %d", len(d.Labels()), d.NumRow())
	}
}

func TestDetectFileFormatExtFallback(t *testing.T) {
	path := filepath.Join("..", "testdata", "csrmat.libsvm")
	if got := DetectFileFormat(path); got != FormatLIBSVM {
		t.Fatalf("DetectFileFormat=%v want LIBSVM", got)
	}
}

func TestSniffCSVWithHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train.csv")
	content := "feat0,feat1,label\n1,2,0\n3,4,1\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	sniff, err := SniffFileFormat(path)
	if err != nil {
		t.Fatal(err)
	}
	if sniff.Format != FormatCSV {
		t.Fatalf("format=%v want CSV", sniff.Format)
	}
	if !sniff.CSV.HasHeader || !sniff.CSV.HasLabelColumn {
		t.Fatalf("csv opts=%+v", sniff.CSV)
	}
	m, err := FromFileAuto(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.NumRow() != 2 || m.NumCol() != 2 {
		t.Fatalf("shape %dx%d", m.NumRow(), m.NumCol())
	}
}

func TestFromFileExplicitFormatOverridesSniff(t *testing.T) {
	path := filepath.Join("..", "testdata", "breast_cancer_train.tsv")
	m, err := FromFile(path, FileLoadOptions{
		Format: FormatTSVLabelLast,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.NumRow() == 0 {
		t.Fatal("empty")
	}
}
