# HTTP 推理示例

非官方 serving；演示如何在自有服务中 embed `leaves` 批预测。

> 需要**可拆独立仓**的脚手架（优雅退出、max batch、/ready、热加载演示、Dockerfile）见  
> [`../serving-template`](../serving-template)（LIB-30）。本目录保持最小单文件 demo。

## 启动

```bash
cd examples/http
export LEAVES_MODEL=../../testdata/xgboost_smoke.json
go run .
# 默认 :8080，可用 LEAVES_HTTP_ADDR 覆盖
```

## API

### `GET /health`

返回 `ok`。

### `POST /predict`

**单条**（`features` 为一维特征向量）：

```json
{"features": [0,1,0,1,0,1,0,1]}
```

响应：`{"prediction": 0.42}`

**批预测**（`rows` 或 `batch`，每行一条样本）：

```json
{
  "rows": [
    [0,1,0,1,0,1,0,1],
    [1,0,1,0,1,0,1,0]
  ]
}
```

响应：`{"predictions": [0.42, 0.38], "nrows": 2}`

**扁平矩阵**（`features` + `nrows` / `ncols`）：

```json
{
  "features": [0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0],
  "nrows": 2,
  "ncols": 8
}
```

```bash
curl -s localhost:8080/predict -d '{"rows":[[0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0]]}'
```
