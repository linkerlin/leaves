// Package metrics 实现评估指标，注册式扩展。
//
// Resolve 按名解析（支持 ndcg@K / map@K 前缀拆 K），Evaluate 计算指标值；
// 内置 rmse / rmsle / logloss / mlogloss / merror / error / auc /
// ndcg / map 等。自定义指标用 Register 注册。
package metrics
