package utils

import (
	"fmt"
	"math/rand"
	"reflect"
	"runtime"

	mn "github.com/made2591/go-perceptron-go/model/neural"
	mu "github.com/made2591/go-perceptron-go/util"
)

func getFunctionName(i interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
}

// MLPtoString retruns a string that represent a mlp
func MLPtoString(mlp *mn.MultiLayerNetwork) string {
	out := ""

	out += fmt.Sprintf("Lrate: %f\nT_func: %s\nT_func_d: %s\n",
		mlp.L_rate, getFunctionName(mlp.T_func), getFunctionName(mlp.T_func_d))

	for i, layer := range mlp.NeuralLayers {
		out += fmt.Sprintf("Layer: %d - Length: %d\n", i, layer.Length)

		for j, neuron := range layer.NeuronUnits {
			out += fmt.Sprintf("\tNeuron: %d - Bias: %f - Delta: %f - Lrate: %f - Value: %f\n\t\tWeights (#%d): %v\n",
				j, neuron.Bias, neuron.Delta, neuron.Lrate, neuron.Value, len(neuron.Weights), neuron.Weights)
		}
	}

	return out
}

// PredictN the output for a set of patterns
func PredictN(mlp *mn.MultiLayerNetwork, input []mn.Pattern) (out [][]float64) {
	out = make([][]float64, len(input))

	for i, pattern := range input {
		out[i] = mn.Execute(mlp, &pattern)
	}

	return out
}

// RoundPrediction puts all predictions to 0 except the one with higher probability
func RoundPrediction(prediction []float64) int {
	_, index := mu.MaxInSlice(prediction)

	return index
}

// RoundPredictions rounds all predictions
func RoundPredictions(predictions [][]float64) []float64 {
	roundPredictions := make([]float64, len(predictions))

	for i, prediction := range predictions {
		roundPredictions[i] = float64(RoundPrediction(prediction))
	}

	return roundPredictions
}

// AccuracyN gets the accuracy of the model's predictions
func AccuracyN(roundedPredictions []float64, actual []mn.Pattern) (int, float64) {
	expected := make([]float64, len(actual))

	for i, pattern := range actual {
		expected[i] = pattern.SingleExpectation
	}

	return mn.Accuracy(expected, roundedPredictions)
}

// RandIntInRange returns a random integer in range [min, max)
func RandIntInRange(min, max int, rng *rand.Rand) int {
	return rng.Intn(max-min) + min
}

// RandFloatInRange returns a float integer in range [min, max]
func RandFloatInRange(min, max float64, rng *rand.Rand) float64 {
	return (max-min)*rng.Float64() + min
}

// RemoveNeuron the given index i from the slice of neurons
func RemoveNeuron(slice []mn.NeuronUnit, i int) []mn.NeuronUnit {
	return append(slice[:i], slice[i+1:]...)
}

// RemoveF64 the given index i from the slice of floats
func RemoveF64(slice []float64, i int) []float64 {
	return append(slice[:i], slice[i+1:]...)
}
