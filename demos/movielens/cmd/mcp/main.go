// MovieLens ranker MCP server（stdio JSON-RPC，供 Agent 调用）。
//
// 配置示例（Cursor / Claude Desktop / 兼容客户端）：
//
//	{
//	  "mcpServers": {
//	    "leaves-movielens": {
//	      "command": "go",
//	      "args": ["run", "./demos/movielens/cmd/mcp"],
//	      "cwd": "/path/to/leaves"
//	    }
//	  }
//	}
//
// 协议：MCP tools/list + tools/call 子集（JSON-RPC 2.0 over stdin/stdout）。
// 注意：日志只能写 stderr，stdout 专供 JSON-RPC。
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/linkerlin/leaves/demos/movielens/agentops"
)

const protocolVersion = "2024-11-05"

type rpcReq struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
}

type rpcResp struct {
	JSONRPC string  `json:"jsonrpc"`
	ID      any     `json:"id,omitempty"`
	Result  any     `json:"result,omitempty"`
	Error   *rpcErr `json:"error,omitempty"`
}

type rpcErr struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func main() {
	// 人类误跑：打印帮助到 stderr
	if len(os.Args) > 1 && (os.Args[1] == "-h" || os.Args[1] == "--help") {
		fmt.Fprintln(os.Stderr, "leaves MovieLens MCP server — speak JSON-RPC on stdin")
		fmt.Fprintln(os.Stderr, "See demos/movielens/TUTORIAL.md §Agent + MCP")
		os.Exit(0)
	}
	br := bufio.NewReader(os.Stdin)
	for {
		line, err := br.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				return
			}
			fmt.Fprintf(os.Stderr, "read: %v\n", err)
			return
		}
		line = trimSpace(line)
		if len(line) == 0 {
			continue
		}
		var req rpcReq
		if err := json.Unmarshal(line, &req); err != nil {
			writeResp(rpcResp{JSONRPC: "2.0", Error: &rpcErr{Code: -32700, Message: "parse error"}})
			continue
		}
		handle(req)
	}
}

func trimSpace(b []byte) []byte {
	i, j := 0, len(b)
	for i < j && (b[i] == ' ' || b[i] == '\t' || b[i] == '\r' || b[i] == '\n') {
		i++
	}
	for j > i && (b[j-1] == ' ' || b[j-1] == '\t' || b[j-1] == '\r' || b[j-1] == '\n') {
		j--
	}
	return b[i:j]
}

func handle(req rpcReq) {
	// notifications：无 id
	if req.ID == nil && req.Method == "notifications/initialized" {
		return
	}
	switch req.Method {
	case "initialize":
		writeResp(rpcResp{
			JSONRPC: "2.0", ID: req.ID,
			Result: map[string]any{
				"protocolVersion": protocolVersion,
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "leaves-movielens-ranker", "version": "1.0.0"},
			},
		})
	case "ping":
		writeResp(rpcResp{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{}})
	case "tools/list":
		writeResp(rpcResp{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{"tools": toolDefs()}})
	case "tools/call":
		var p struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		_ = json.Unmarshal(req.Params, &p)
		res := callTool(p.Name, p.Arguments)
		text, _ := json.MarshalIndent(res, "", "  ")
		writeResp(rpcResp{
			JSONRPC: "2.0", ID: req.ID,
			Result: map[string]any{
				"content": []map[string]any{
					{"type": "text", "text": string(text)},
				},
				"isError": !res.OK,
			},
		})
	default:
		writeResp(rpcResp{
			JSONRPC: "2.0", ID: req.ID,
			Error: &rpcErr{Code: -32601, Message: "method not found: " + req.Method},
		})
	}
}

func writeResp(r rpcResp) {
	b, _ := json.Marshal(r)
	os.Stdout.Write(append(b, '\n'))
}

func toolDefs() []map[string]any {
	return []map[string]any{
		tool("movielens_status", "检查 MovieLens ranking 数据与模型是否就绪", map[string]any{
			"type": "object", "properties": map[string]any{},
		}),
		tool("movielens_prepare", "准备/生成 rank_movielens_*.tsv（可 force 重跑 gen 脚本）", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"force": map[string]any{"type": "boolean", "description": "强制重新生成"},
			},
		}),
		tool("movielens_train", "训练 leaves ranker（rank:ndcg/pairwise/listwise）", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"objective": map[string]any{"type": "string", "description": "rank:ndcg|rank:pairwise|rank:listwise"},
				"rounds":    map[string]any{"type": "integer"},
				"depth":     map[string]any{"type": "integer"},
				"lr":        map[string]any{"type": "number"},
				"out_model": map[string]any{"type": "string"},
				"metrics":   map[string]any{"type": "string"},
			},
		}),
		tool("movielens_eval", "在测试集上评估 NDCG@k", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"model":  map[string]any{"type": "string"},
				"ndcg_k": map[string]any{"type": "integer"},
			},
		}),
		tool("movielens_recommend", "对测试用户输出 Top-K 推荐列表（JSON）", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"model": map[string]any{"type": "string"},
				"group": map[string]any{"type": "integer", "description": "测试用户序号 0..14"},
				"qid":   map[string]any{"type": "integer"},
				"topk":  map[string]any{"type": "integer"},
			},
		}),
		tool("movielens_full_pipeline", "一键：prepare→train→eval→recommend（精排-only）", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"objective": map[string]any{"type": "string"},
				"group":     map[string]any{"type": "integer"},
				"topk":      map[string]any{"type": "integer"},
			},
		}),
		tool("movielens_four_stage", "MovieLens 四段：prep→召回100→LTR→发牌（Tag 控重）", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"workspace":    map[string]any{"type": "string"},
				"train_users":  map[string]any{"type": "integer"},
				"test_users":   map[string]any{"type": "integer"},
				"recall_size":  map[string]any{"type": "integer"},
				"rounds":       map[string]any{"type": "integer"},
				"deck_size":    map[string]any{"type": "integer"},
				"max_same_tag": map[string]any{"type": "integer"},
			},
		}),
	}
}

func tool(name, desc string, schema map[string]any) map[string]any {
	return map[string]any{
		"name":        name,
		"description": desc,
		"inputSchema": schema,
	}
}

func callTool(name string, args map[string]any) agentops.Result {
	if args == nil {
		args = map[string]any{}
	}
	switch name {
	case "movielens_status":
		return agentops.Status()
	case "movielens_prepare":
		force, _ := args["force"].(bool)
		return agentops.Prepare(force)
	case "movielens_train":
		p := agentops.TrainParams{}
		if s, ok := args["objective"].(string); ok {
			p.Objective = s
		}
		p.Rounds = asInt(args["rounds"])
		p.Depth = asInt(args["depth"])
		if v, ok := args["lr"].(float64); ok {
			p.LR = v
		}
		if s, ok := args["out_model"].(string); ok {
			p.OutModel = s
		}
		if s, ok := args["metrics"].(string); ok {
			p.Metrics = s
		}
		return agentops.Train(p)
	case "movielens_eval":
		model, _ := args["model"].(string)
		k := asInt(args["ndcg_k"])
		return agentops.Eval(model, k)
	case "movielens_recommend":
		p := agentops.RecommendParams{QID: -1}
		if s, ok := args["model"].(string); ok {
			p.Model = s
		}
		p.Group = asInt(args["group"])
		if v, ok := args["qid"]; ok {
			p.QID = asInt(v)
		}
		p.TopK = asInt(args["topk"])
		return agentops.Recommend(p)
	case "movielens_full_pipeline":
		tp := agentops.TrainParams{}
		if s, ok := args["objective"].(string); ok {
			tp.Objective = s
		}
		rp := agentops.RecommendParams{Group: asInt(args["group"]), TopK: asInt(args["topk"]), QID: -1}
		return agentops.FullPipeline(tp, rp)
	case "movielens_four_stage":
		p := agentops.FourStageParams{}
		if s, ok := args["workspace"].(string); ok {
			p.Workspace = s
		}
		p.TrainUsers = asInt(args["train_users"])
		p.TestUsers = asInt(args["test_users"])
		p.RecallSize = asInt(args["recall_size"])
		p.Rounds = asInt(args["rounds"])
		p.DeckSize = asInt(args["deck_size"])
		p.MaxSameTag = asInt(args["max_same_tag"])
		return agentops.FourStage(p)
	default:
		return agentops.Result{OK: false, Op: name, Error: "unknown tool: " + name,
			Hint: "use tools/list", TS: ""}
	}
}

func asInt(v any) int {
	switch x := v.(type) {
	case float64:
		return int(x)
	case int:
		return x
	case int64:
		return int(x)
	case json.Number:
		i, _ := x.Int64()
		return int(i)
	default:
		return 0
	}
}
