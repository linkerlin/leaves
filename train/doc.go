// Package train 实现 GBRT 训练流水线。
//
// 入口：NewLearner 构造训练器，Learner.Fit 训练，CrossValidate 交叉验证。
// 目标函数与评估指标分别由 objective / metrics 包的注册表提供。
// 支持 hist/exact/gpu_hist 树方法、排序/survival/tweedie 目标、
// checkpoint 续训（ResumeFit）与外存矩阵（FitExternal）。
package train
