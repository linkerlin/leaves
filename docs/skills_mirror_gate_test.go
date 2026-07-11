package docs_test

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestSkillsMirrorSync 锁定 POST-01：skills/ 与 .cursor/skills/ 内容一致，
// 防止 Cursor 发现入口与 canonical SKILL 双份漂移（演进方案 §10.3 / WP-06）。
func TestSkillsMirrorSync(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller")
	}
	root := filepath.Join(filepath.Dir(file), "..")
	srcRoot := filepath.Join(root, "skills")
	dstRoot := filepath.Join(root, ".cursor", "skills")

	if st, err := os.Stat(srcRoot); err != nil || !st.IsDir() {
		t.Fatalf("skills/ missing: %v", err)
	}
	if st, err := os.Stat(dstRoot); err != nil || !st.IsDir() {
		t.Fatalf(".cursor/skills/ missing: %v — 请从 skills/ 同步镜像", err)
	}

	// 以 skills/ 为真源：每个文件必须在 .cursor/skills 有相同内容。
	var drifts []string
	err := filepath.WalkDir(srcRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		// 忽略编辑器垃圾
		base := d.Name()
		if strings.HasPrefix(base, ".") || strings.HasSuffix(base, "~") {
			return nil
		}
		rel, err := filepath.Rel(srcRoot, path)
		if err != nil {
			return err
		}
		mirror := filepath.Join(dstRoot, rel)
		if _, err := os.Stat(mirror); err != nil {
			drifts = append(drifts, "missing mirror: "+filepath.ToSlash(rel))
			return nil
		}
		h1, err := fileSHA256(path)
		if err != nil {
			return err
		}
		h2, err := fileSHA256(mirror)
		if err != nil {
			return err
		}
		if h1 != h2 {
			drifts = append(drifts, "content drift: "+filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk skills/: %v", err)
	}

	// 反向：.cursor 多出的文件也报（避免只维护镜像侧）。
	err = filepath.WalkDir(dstRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		base := d.Name()
		if strings.HasPrefix(base, ".") || strings.HasSuffix(base, "~") {
			return nil
		}
		rel, err := filepath.Rel(dstRoot, path)
		if err != nil {
			return err
		}
		canon := filepath.Join(srcRoot, rel)
		if _, err := os.Stat(canon); err != nil {
			drifts = append(drifts, "orphan in .cursor/skills (no skills/ source): "+filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk .cursor/skills/: %v", err)
	}

	if len(drifts) > 0 {
		t.Fatalf("skills mirror out of sync (%d):\n  - %s\n\nfix: copy skills/<name>/ → .cursor/skills/<name>/ (canonical = skills/)",
			len(drifts), strings.Join(drifts, "\n  - "))
	}
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
