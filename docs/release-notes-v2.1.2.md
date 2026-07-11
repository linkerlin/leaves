# leaves v2.1.2

**主题**：Explain 性能（LIB-22）、可拆 serving 模板（LIB-30）、文档/回归门禁加固。

## Highlights

1. **Tree SHAP 更快、更少分配**：`TreeExplainer` 预计算节点覆盖权重与背景 margin，多样本复用 path 缓冲。
2. **Serving 脚手架**：[`examples/serving-template`](../examples/serving-template/) 可整目录复制为独立服务（非官方 serving 框架）。
3. **质量门禁**：sklearn 实验边界写实 + `testdata-matrix.md` 工件存在性测试。

## Recommended API

- Infer: `LoadFromFile` + `DefaultLoadOptions`
- Explain: `explain.NewTreeExplainer(forest).ShapleyValues(rows)`
- Embed HTTP: `examples/http`（最小）或 `examples/serving-template`（可拆仓）

## Compatibility

- SHAP **数值语义不变**（既有 golden / 可加性测试仍绿）
- metrics/CLI schema：**无破坏**；`schema_version` 仍为 `1`
- Breaking: **无**

## CI / 本地

```powershell
go test ./explain -count=1
go test ./explain -bench=TreeSHAP -benchmem -count=1
go test ./examples/serving-template/... -count=1
go test ./docs -run 'TestdataMatrix|SkillsMirror' -count=1
```

## Docs

- [CHANGELOG](../CHANGELOG.md) · [benchmark-baseline](benchmark-baseline.md) · [serving-template README](../examples/serving-template/README.md)
