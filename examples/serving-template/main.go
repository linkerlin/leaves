// leaves HTTP serving 模板（LIB-30）。
//
// 设计目标：可整目录复制为独立仓库；本仓 examples/http 仍为最小 embed demo。
// leaves 库本身不做官方 serving 框架。
//
//	LEAVES_MODEL=../../testdata/xgboost_smoke.json go run .
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/example/leaves-serving/internal/config"
	"github.com/example/leaves-serving/internal/handler"
	"github.com/example/leaves-serving/internal/modelhost"
)

func main() {
	cfg, err := config.FromEnv()
	if err != nil {
		// 开发便利：未设 LEAVES_MODEL 时尝试 monorepo 相对路径
		if os.Getenv("LEAVES_MODEL") == "" {
			fallback := "../../testdata/xgboost_smoke.json"
			if _, e := os.Stat(fallback); e == nil {
				_ = os.Setenv("LEAVES_MODEL", fallback)
				cfg, err = config.FromEnv()
			}
		}
		if err != nil {
			log.Fatalf("config: %v", err)
		}
	}

	host, err := modelhost.Load(cfg.ModelPath)
	if err != nil {
		log.Fatalf("model: %v", err)
	}
	defer host.Close()

	api := &handler.API{
		Host:      host,
		MaxBatch:  cfg.MaxBatch,
		StartedAt: time.Now(),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", api.Health)
	mux.HandleFunc("/ready", api.Ready)
	mux.HandleFunc("/meta", api.Meta)
	mux.HandleFunc("/metrics", api.Metrics)
	mux.HandleFunc("/predict", api.Predict)
	// 演示热加载：生产务必加鉴权 / 内网
	mux.HandleFunc("/admin/reload", api.Reload)

	srv := &http.Server{
		Addr:         cfg.Addr,
		Handler:      mux,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}

	go func() {
		log.Printf("leaves-serving on %s model=%s max_batch=%d", cfg.Addr, cfg.ModelPath, cfg.MaxBatch)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	log.Printf("shutting down (grace=%s)…", cfg.ShutdownGrace)
	ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGrace)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}
