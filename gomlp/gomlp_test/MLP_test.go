package gomlp_test

import (
	"fmt"
	"testing"

	gomlp "github.com/salvacorts/TFG-Parasitic-Metaheuristics/gomlp"
	transf "github.com/salvacorts/TFG-Parasitic-Metaheuristics/gomlp/transferfunctions"
)

func TestBasicMLP(t *testing.T) {
	layers := []int{4, 5, 2}
	sigm := transf.MakeSigmoidalTransferFunc()

	mlp := gomlp.MakeMLP(layers, 0.6, sigm)

	// Crate testing data
	inputs := make([]float64, 4)
	outputs := make([]float64, 2)

	// Train
	accuracy := mlp.Train(inputs, outputs)
	t.Logf("Error: %f", accuracy)

	// Predict
	inputs = make([]float64, 4)
	outputs = mlp.Predict(inputs)
	t.Logf("Outputs: %v", outputs)
}

func main() {
	layers := []int{4, 5, 2}
	sigm := transf.MakeSigmoidalTransferFunc()

	mlp := gomlp.MakeMLP(layers, 0.6, sigm)

	// Crate testing data
	inputs := make([]float64, 4)
	outputs := make([]float64, 2)

	// Train
	accuracy := mlp.Train(inputs, outputs)
	fmt.Printf("Error: %f", accuracy)

	// Predict
	inputs = make([]float64, 4)
	outputs = mlp.Predict(inputs)
	fmt.Printf("Outputs: %v", outputs)
}
