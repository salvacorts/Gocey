package common

import (
	"math/rand"

	mn "github.com/made2591/go-perceptron-go/model/neural"
	"github.com/salvacorts/eaopt"
)

// TransferFunction stands for a transfer function
type TransferFunction = func(float64) float64

// MLPFactory is a type to create a factory of MLP randomly initialized
type MLPFactory struct {
	// Network topology
	InputLayers     int
	OutputLayers    int
	MinHiddenLayers int
	MaxHiddenLayers int
	Tfunc           TransferFunction
	TfuncDeriv      TransferFunction

	// Training Info configuration
	MaxLR  float64
	MinLR  float64
	Config MLPConfig
}

// NewRandMLP creates a randomly initialized MLP
func (f MLPFactory) NewRandMLP(rng *rand.Rand) eaopt.Genome {
	layers := []int{
		f.InputLayers,
		f.MinHiddenLayers + rng.Int()%f.MaxHiddenLayers,
		f.OutputLayers}

	learningRate := f.MaxLR + rng.Float64()*(f.MaxLR-f.MinLR)

	return MLP{
		NeuralNet: mn.PrepareMLPNet(layers, learningRate, f.Tfunc, f.TfuncDeriv),
		Config:    f.Config,
	}
}
