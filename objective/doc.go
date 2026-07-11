// Package objective 实现训练目标函数：损失及其一阶（grad）/二阶（hess）梯度。
//
// 注册式扩展：Register 注册自定义目标，ByNameWithClass 按名（含 num_class）解析；
// 内置 binary:logistic / reg:squarederror / multi:* / rank:ndcg /
// survival:cox / survival:aft / reg:tweedie 等。
package objective
