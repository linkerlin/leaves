package objective

func init() {
	Register("reg:squarederror", func(int) (Func, error) { return SquaredError{}, nil })
	Register("", func(int) (Func, error) { return SquaredError{}, nil })
	Register("binary:logistic", func(int) (Func, error) { return BinaryLogistic{}, nil })
	Register("reg:gamma", func(int) (Func, error) { return Gamma{}, nil })
	Register("count:poisson", func(int) (Func, error) { return Poisson{}, nil })
	Register("reg:tweedie", func(int) (Func, error) { return NewTweedie(defaultTweediePower), nil })
	Register("survival:cox", func(int) (Func, error) { return Cox{}, nil })
	Register("survival:aft", func(int) (Func, error) { return AFTNormal{}, nil })
}
