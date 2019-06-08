package gomlp

import (
	"fmt"

	transf "github.com/salvacorts/TFG-Parasitic-Metaheuristics/gomlp/transferfunctions"
)

// MLP implements a Multi-Layer Perceptron
type MLP struct {
	LearningRate float64
	Layers       []Layer
	TransferFunc transf.TransferFunction
}

// MakeMLP initialized a MLP object
func MakeMLP(layers []int, learningRate float64, transfterFunction transf.TransferFunction) MLP {
	mlp := MLP{
		LearningRate: learningRate,
		Layers:       make([]Layer, len(layers)),
		TransferFunc: transfterFunction}

	// Create layers with the given topology
	for i := 0; i < len(layers); i++ {
		if i != 0 {
			mlp.Layers[i] = MakeLayer(layers[i], layers[i-1])
		} else {
			mlp.Layers[i] = MakeLayer(layers[i], 0) // The first layer is not connected to a previous one
		}
	}

	return mlp
}

// Train the model with a set of examples
func (mlp MLP) Train(input [][]float64, output [][]float64) {
	if (len(input) < 1 && len(output) < 1) || len(input[0]) < 1 {
		fmt.Printf("[!] Malformed training data")
		return
	}

	for i := 0; i < len(input); i++ {
		mlp.backPropagation(input[i], output[i])
	}
}

// Train trains the mlp with backpropagation (Singlethread)
func (mlp MLP) backPropagation(input []float64, output []float64) {
	newOutput := mlp.Predict(input)

	// Calculate output deltas with output error
	for i := 0; i < len(mlp.Layers[len(mlp.Layers)-1].Neurons); i++ {
		error := output[i] - newOutput[i]
		mlp.Layers[len(mlp.Layers)-1].Neurons[i].Delta =
			error + mlp.TransferFunc.EvaluateDerivate(newOutput[i])
	}

	// Backpropagate the error
	for i := len(mlp.Layers) - 2; i >= 0; i-- {

		// Calculte error based on the following layer
		for j := 0; j < len(mlp.Layers[i].Neurons); j++ { // For each neuron in the layer
			error := 0.0

			for k := 0; k < len(mlp.Layers[i+1].Neurons); k++ { // For each neuron in the next layer
				error += mlp.Layers[i+1].Neurons[k].Delta * mlp.Layers[i+1].Neurons[k].Weights[j]
			}

			mlp.Layers[i].Neurons[j].Delta = error * mlp.TransferFunc.EvaluateDerivate(
				mlp.Layers[i].Neurons[j].Value)
		}

		// Update weight of the following layer
		for j := 0; j < len(mlp.Layers[i+1].Neurons); j++ { // for each neuron in the next layer
			for k := 0; k < len(mlp.Layers[i].Neurons); k++ { // for each neuron in the current layer
				mlp.Layers[i+1].Neurons[j].Weights[k] +=
					mlp.LearningRate * mlp.Layers[i+1].Neurons[j].Delta * mlp.Layers[i].Neurons[k].Value
			}

			mlp.Layers[i+1].Neurons[j].Bias += mlp.LearningRate * mlp.Layers[i+1].Neurons[j].Delta
		}

	}
}

// Predict takes an input of features and predict an output
func (mlp MLP) Predict(input []float64) []float64 {
	outputSize := len(mlp.Layers[len(mlp.Layers)-1].Neurons) // Number of neurons in the last layer
	output := make([]float64, outputSize)

	// Put inputs in the input layer
	for i := 0; i < len(mlp.Layers[0].Neurons); i++ {
		mlp.Layers[0].Neurons[i].Value = input[i]
	}

	// Iterate thorugh every neuron of every hidden and output layers
	for i := 1; i < len(mlp.Layers); i++ {
		for j := 0; j < len(mlp.Layers[i].Neurons); j++ {
			value := mlp.Layers[i].Neurons[j].Bias

			for k := 0; k < len(mlp.Layers[i-1].Neurons); k++ {
				value += mlp.Layers[i].Neurons[j].Weights[k] * mlp.Layers[i-1].Neurons[k].Value
			}

			mlp.Layers[i].Neurons[j].Value = mlp.TransferFunc.Evaluate(value)
		}
	}

	// Get output
	for i := 0; i < outputSize; i++ {
		output[i] = mlp.Layers[len(mlp.Layers)-1].Neurons[i].Value
	}

	return output
}

// PredictN predicts various instances
func (mlp MLP) PredictN(inputs [][]float64) [][]float64 {
	predictions := make([][]float64, len(inputs))

	for i, x := range inputs {
		predictions[i] = mlp.Predict(x)
	}

	return predictions
}
