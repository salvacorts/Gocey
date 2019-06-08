package transferfunctions

import "math"

// SigmoidalTransferFunc is a type that implements sigmoidal transfer function
type SigmoidalTransferFunc struct{}

// MakeSigmoidalTransferFunc creates a SigmoidalTransferFunc
func MakeSigmoidalTransferFunc() SigmoidalTransferFunc {
	return SigmoidalTransferFunc{}
}

// Evaluate implements TransferFunction Evaluate method using a sigmoid function
func (f SigmoidalTransferFunc) Evaluate(value float64) float64 {
	return 1 / (1 + math.Pow(math.E, -value))
}

// EvaluateDerivate implements TransferFunction EvaluateDerivate method using a derivated sigmoid function
func (f SigmoidalTransferFunc) EvaluateDerivate(value float64) float64 {
	return (value - math.Pow(value, 2))
}
