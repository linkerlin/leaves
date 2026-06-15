package data

import (
	"path/filepath"
	"testing"
)

func TestFromLIBSVM(t *testing.T) {
	path := filepath.Join("..", "testdata", "csrmat.libsvm")
	csr, err := FromLIBSVM(path, LIBSVMOptions{HasLabel: true})
	if err != nil {
		t.Fatal(err)
	}
	if csr.NumRow() == 0 || csr.NumCol() == 0 {
		t.Fatalf("empty matrix %dx%d", csr.NumRow(), csr.NumCol())
	}
	if len(csr.Labels()) != csr.NumRow() {
		t.Fatalf("labels %d != rows %d", len(csr.Labels()), csr.NumRow())
	}
}

func TestFromFileLIBSVM(t *testing.T) {
	path := filepath.Join("..", "testdata", "csrmat.libsvm")
	m, err := FromFile(path, FileLoadOptions{
		Format: FormatLIBSVM,
		LIBSVM: LIBSVMOptions{HasLabel: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.NumRow() == 0 {
		t.Fatal("empty")
	}
}
