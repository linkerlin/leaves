package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config 服务配置（环境变量驱动，便于 K8s/Compose）。
type Config struct {
	// Addr 监听地址，默认 :8080
	Addr string
	// ModelPath leaves 可加载模型（leaves.json / XGB JSON 等）
	ModelPath string
	// MaxBatch 单次 /predict 最大样本数
	MaxBatch int
	// ReadTimeout / WriteTimeout HTTP 超时
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	// ShutdownGrace 优雅退出等待
	ShutdownGrace time.Duration
}

// FromEnv 读取配置。
func FromEnv() (Config, error) {
	cfg := Config{
		Addr:          envOr("LEAVES_HTTP_ADDR", ":8080"),
		ModelPath:     envOr("LEAVES_MODEL", ""),
		MaxBatch:      envInt("LEAVES_MAX_BATCH", 4096),
		ReadTimeout:   envDuration("LEAVES_READ_TIMEOUT", 10*time.Second),
		WriteTimeout:  envDuration("LEAVES_WRITE_TIMEOUT", 30*time.Second),
		ShutdownGrace: envDuration("LEAVES_SHUTDOWN_GRACE", 10*time.Second),
	}
	if cfg.ModelPath == "" {
		return cfg, fmt.Errorf("LEAVES_MODEL is required (path to leaves.json / XGB JSON / …)")
	}
	if cfg.MaxBatch < 1 {
		return cfg, fmt.Errorf("LEAVES_MAX_BATCH must be >= 1")
	}
	return cfg, nil
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func envDuration(k string, def time.Duration) time.Duration {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return def
	}
	return d
}
