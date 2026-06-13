# 项目长期记忆

## 项目信息
- **项目名称**: leaves — 纯 Go 语言 GBRT 预测与训练库
- **上游**: github.com/dmitryikh/leaves (当前版本 v0.8.0)
- **计算底座**: **Born** (github.com/born-ml/born，本地 C:\GitHub\born)
- **已废弃**: GoMLX、gogpu 直连（2026-06-13 决策）
- **参考**: XGBoost C++ 项目 (C:\GitHub\xgboost\)

## 用户偏好
- 使用中文思考和输出
- Windows 原生，不用 WSL
- 以费曼风格解释问题：直指本质、具体例子、简单语言

## 架构约定
- 训练产出 `leaves.json`；推理 `ModelIR` → `tree.Engine`
- `train/` 依赖 `tree/`，`tree/` 不依赖 `train/`
- NativeEngine 为正确性 golden；BornEngine 为加速路径

## 演进计划
- `演进计划.md`：v3.0 Born 迁移版
