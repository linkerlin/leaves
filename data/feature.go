package data

// FeatureType 特征类型。
type FeatureType int

const (
	FeatureNumeric FeatureType = iota
	FeatureCategorical
)

// FeatureTypesMatrix 带特征类型元信息的矩阵。
type FeatureTypesMatrix interface {
	Matrix
	FeatureTypes() []FeatureType
}

// FeatureNamesMatrix 带特征名的矩阵。
type FeatureNamesMatrix interface {
	Matrix
	FeatureNames() []string
}

// IsCategorical 判断特征列是否为分类。
func IsCategorical(dm Matrix, feat int) bool {
	ft, ok := dm.(FeatureTypesMatrix)
	if !ok {
		return false
	}
	types := ft.FeatureTypes()
	if feat < 0 || feat >= len(types) {
		return false
	}
	return types[feat] == FeatureCategorical
}
