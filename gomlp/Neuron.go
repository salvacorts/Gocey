package gomlp

import "math/rand"

// Neuron is a struct that contains a neuron information
type Neuron struct {
	Value   float64
	Weights []float64
	Bias    float64
	Delta   float64
}

// MakeNeuron is a Neuron constructor
func MakeNeuron(previousLayerSize int) Neuron {
	neuron := Neuron{
		Value:   rand.Float64(),
		Weights: make([]float64, previousLayerSize),
		Bias:    rand.Float64(),
		Delta:   rand.Float64()}

	for i := 0; i < len(neuron.Weights); i++ {
		neuron.Weights[i] = rand.Float64()
	}

	return neuron
}
