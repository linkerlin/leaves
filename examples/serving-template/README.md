# leaves serving 模板（LIB-30）

> **非官方 serving 框架**。本目录是可整包拆出的独立服务脚手架；  
> leaves 主库只提供推理 API，官方边界见 [TODO 明确不做](../../TODO.md)。  
> 仓内最小 embed 演示仍见 [`../http`](../http)。

## 与 `examples/http` 的分工

| | `examples/http` | **本模板** |
|--|-----------------|------------|
| 目标 | 最小 embed 演示 | 可复制为独立服务 |
| 结构 | 单 `main.go` | `config` / `modelhost` / `handler` 分包 |
| 运维 | 无 | 优雅退出、max batch、/ready、热加载演示、Dockerfile |
| go.mod | 用根模块 | **独立 module** + `replace`（可拆仓） |

## 本地运行（monorepo）

```powershell
cd examples/serving-template
$env:LEAVES_MODEL = "../../testdata/xgboost_smoke.json"
go mod tidy
go run .
```

```powershell
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/meta
curl -s http://localhost:8080/predict -H "Content-Type: application/json" `
  -d "{\"rows\":[[0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0]]}"
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活 |
| GET | `/ready` | 模型已加载 |
| GET | `/meta` | `n_features` / `n_outputs` / 模型路径 |
| GET | `/metrics` | 简易请求计数（演示） |
| POST | `/predict` | 单条 / 批 / 扁平矩阵（同 http demo JSON） |
| POST | `/admin/reload` | `{"path":"..."}` 热替换模型（**无鉴权，仅演示**） |

多输出模型：`predictions` 为扁平 `nrows * n_outputs`，并附 `matrix`（每行一个向量）。

## 拆成独立仓库

```text
1. 复制 examples/serving-template/ 全部文件到新 git 仓库
2. 编辑 go.mod：
   - module 改成你的路径
   - 删除 replace 行
   - require github.com/linkerlin/leaves/v2 v2.1.1  （或更新版本）
3. go mod tidy && go test ./...
4. 生产：加鉴权、TLS、限流、结构化日志、Prometheus
```

## Docker

```powershell
# monorepo 构建较特殊：Dockerfile 假定上下文为模板目录；
# 独立仓最简单：在独立仓根目录 docker build -t leaves-serving .
docker build -t leaves-serving .
docker run --rm -p 8080:8080 -v /path/to/models:/models `
  -e LEAVES_MODEL=/models/model.leaves.json leaves-serving
```

## 环境变量

见 [`.env.example`](.env.example)。

| 变量 | 默认 | 含义 |
|------|------|------|
| `LEAVES_MODEL` | （必填） | 模型路径 |
| `LEAVES_HTTP_ADDR` | `:8080` | 监听 |
| `LEAVES_MAX_BATCH` | `4096` | 单次最大样本数 |
| `LEAVES_READ_TIMEOUT` | `10s` | |
| `LEAVES_WRITE_TIMEOUT` | `30s` | |
| `LEAVES_SHUTDOWN_GRACE` | `10s` | 优雅退出 |

## 测试

```powershell
cd examples/serving-template
go test ./... -count=1
```

## 明确不做（留给你的平台）

- 鉴权 / mTLS / API Key  
- gRPC / 自动扩缩  
- 模型 registry 拉取（可用 `leaves publish` 本地包 + 你的 CI）  
- 把本服务并进 `cmd/leaves` 主 CLI  
