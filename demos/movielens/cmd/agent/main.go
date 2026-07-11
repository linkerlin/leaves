// Agent 友好 CLI：每条命令向 stdout 打印一条 JSON Result（供 shell Agent / MCP 封装复用）。
//
//	go run ./demos/movielens/cmd/agent status
//	go run ./demos/movielens/cmd/agent prepare
//	go run ./demos/movielens/cmd/agent train -objective rank:ndcg
//	go run ./demos/movielens/cmd/agent eval
//	go run ./demos/movielens/cmd/agent recommend -group 0 -topk 10
//	go run ./demos/movielens/cmd/agent full-pipeline
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/linkerlin/leaves/demos/movielens/agentops"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd := os.Args[1]
	args := os.Args[2:]
	var res agentops.Result
	switch cmd {
	case "status", "help", "-h", "--help":
		if cmd != "status" {
			usage()
			return
		}
		res = agentops.Status()
	case "prepare":
		fs := flag.NewFlagSet("prepare", flag.ExitOnError)
		force := fs.Bool("force", false, "强制重新生成数据")
		_ = fs.Parse(args)
		res = agentops.Prepare(*force)
	case "train":
		fs := flag.NewFlagSet("train", flag.ExitOnError)
		obj := fs.String("objective", "rank:ndcg", "rank:ndcg|rank:pairwise|rank:listwise")
		rounds := fs.Int("rounds", 0, "轮数")
		depth := fs.Int("depth", 0, "深度")
		lr := fs.Float64("lr", 0, "学习率")
		out := fs.String("out-model", "", "模型路径")
		metrics := fs.String("metrics", "", "metrics.json 路径")
		_ = fs.Parse(args)
		res = agentops.Train(agentops.TrainParams{
			Objective: *obj, Rounds: *rounds, Depth: *depth, LR: *lr,
			OutModel: *out, Metrics: *metrics,
		})
	case "eval":
		fs := flag.NewFlagSet("eval", flag.ExitOnError)
		model := fs.String("model", "", "模型路径")
		k := fs.Int("ndcg-k", 10, "NDCG@k")
		_ = fs.Parse(args)
		res = agentops.Eval(*model, *k)
	case "recommend":
		fs := flag.NewFlagSet("recommend", flag.ExitOnError)
		model := fs.String("model", "", "模型路径")
		group := fs.Int("group", 0, "测试用户序号")
		qid := fs.Int("qid", -1, "qid（覆盖 group）")
		topk := fs.Int("topk", 10, "Top-K")
		out := fs.String("out-json", "", "输出 JSON")
		_ = fs.Parse(args)
		res = agentops.Recommend(agentops.RecommendParams{
			Model: *model, Group: *group, QID: *qid, TopK: *topk, OutJSON: *out,
		})
	case "full-pipeline", "pipeline":
		fs := flag.NewFlagSet("full-pipeline", flag.ExitOnError)
		obj := fs.String("objective", "rank:ndcg", "目标")
		group := fs.Int("group", 0, "推荐用户")
		topk := fs.Int("topk", 10, "Top-K")
		_ = fs.Parse(args)
		res = agentops.FullPipeline(
			agentops.TrainParams{Objective: *obj},
			agentops.RecommendParams{Group: *group, TopK: *topk},
		)
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", cmd)
		usage()
		os.Exit(2)
	}
	agentops.WriteResult(res)
	if !res.OK {
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `用法: agent <command> [flags]

命令:
  status           检查数据/模型是否就绪
  prepare [-force] 生成 MovieLens ranking TSV（调用 gen_rank_movielens.py）
  train   [flags]  训练 ranker → leaves.json + metrics.json
  eval    [flags]  测试集 NDCG
  recommend [flags] Top-K 推荐 JSON
  full-pipeline    prepare→train→eval→recommend

所有成功/失败均向 stdout 打印一条 JSON（Agent 契约）。
MCP 入口: go run ./demos/movielens/cmd/mcp
教程: demos/movielens/TUTORIAL.md`)
}
