package common

import mn "github.com/salvacorts/go-perceptron-go/model/neural"

// Remove the given index i from the slice of neurons
func Remove(slice []NeuronUnit, i int) []NeuronUnit {
	return append(slice[:i], slice[i+1:]...)
}

// PredictN the output for a set of patterns
func PredictN(mlp *MultiLayerNetwork, input []mn.Pattern) (out [][]float64) {
	out = make([][]float64, len(input))

	var (
		tFunc TransferF
	)

	switch mlp.TFunc {
	case TransferFunc_SIGMOIDAL:
		tFunc = mn.SigmoidalTransfer
		// TODO: Add others here
	}

	for i, pattern := range input {
		out[i] = Execute(mlp, &pattern, tFunc)
	}

	return out
}
