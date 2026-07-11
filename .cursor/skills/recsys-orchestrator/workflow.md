# 工作流详表

## 阶段耗时参考（10 万 User，100 Item/User）

| 阶段 | 实现 | 量级 |
|------|------|------|
| 数据准备 | Go 流式 | 分钟级 |
| 召回 | Tag 倒排 | 分钟级 |
| 转 TSV | 批量写 | 秒级 |
| LTR 训练 | leaves hist | 视 feat 维与轮数 |
| 推理 | PredictDense 批 | 秒~分钟 |
| 发牌 | 内存规则 | 秒级 |

## 环境变量

| 变量 | 用途 |
|------|------|
| `LEAVES_TESTDATA` | 指向含 ranking TSV / baseline 的目录 |
| `LEAVES_TRAIN_ACCEL` | 训练加速 auto/cpu/webgpu |

## 测试矩阵

| 数据集 | 脚本 | 测试 |
|--------|------|------|
| smoke | `gen_rank_smoke.py` | `TestRankNDCGTrendVsXGBoost` |
| movielens | `gen_rank_movielens.py` | `TestRankMovieLens` |
| msltr | `gen_rank_msltr.py` | MSLTR 对标测试 |

文档：`docs/testdata-matrix.md`

## 故障排查

| 现象 | 对策 |
|------|------|
| `bad qid` / group 错误 | 同 qid 行须相邻；查 user_qid 序 |
| 特征数不匹配 | catalog 与 rank TSV feat 列对齐 |
| NDCG 低 | 增 NumRound、查 label 分布、扩 feat |
| 发牌不足 10 条 | 放宽 max_same_tag 或扩召回 |
| LoadFromFile 报错 | 勿用 io 加载 TSV；用 data.FromFileAuto |

## 扩展路径

- **增量样本**：追加 `samples_all.tsv` 后重跑 prep+recall
- **热更新模型**：`Ensemble.Reload`（P3 已支持）
- **量化部署**：`quantize/` int8 阈值量化 + parity gate
- **解释性**：`explain/` SHAP / feature importance（非推荐主路径）
