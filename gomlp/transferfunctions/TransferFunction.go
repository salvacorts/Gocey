package transferfunctions

// TransferFunction is an interface used to define transfer functions
type TransferFunction interface {
	Evaluate(value float64) float64
	EvaluateDerivate(value float64) float64
}
