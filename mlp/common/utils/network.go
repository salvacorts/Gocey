package utils

import (
	"math/rand"

	mn "github.com/made2591/go-perceptron-go/model/neural"
	mu "github.com/made2591/go-perceptron-go/util"
)

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

// RandIntInRange returns a random integer between min and max
func RandIntInRange(min, max int, rgn *rand.Rand) int {
	return min + rand.Int()%max
}

// Remove the given index i from the slice
func Remove(slice []mn.NeuronUnit, i int) []mn.NeuronUnit {
	return append(slice[:i], slice[i+1:]...)
}
