package common

import (
	"math/rand"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"

	"github.com/salvacorts/eaopt"
	mn "github.com/salvacorts/go-perceptron-go/model/neural"
)

// TransferFunction stands for a transfer function
type TransferFunction = func(float64) float64

// MLPFactory is a type to create a factory of MLP randomly initialized
type MLPFactory struct {
	// Network topology
	InputLayers      int
	OutputLayers     int
	MinHiddenNeurons int
	MaxHiddenNeurons int
	Tfunc            TransferFunction
	TfuncDeriv       TransferFunction

	// Training Info configuration
	MaxLR  float64
	MinLR  float64
	Config MLPConfig
}

// NewRandMLP creates a randomly initialized MLP
func (f MLPFactory) NewRandMLP(rng *rand.Rand) eaopt.Genome {
	layers := []int{
		f.InputLayers,
		utils.RandIntInRange(f.MinHiddenNeurons, f.MaxHiddenNeurons, rng),
		f.OutputLayers}

	learningRate := f.MaxLR + rng.Float64()*(f.MaxLR-f.MinLR)

	new := mn.PrepareMLPNet(layers, learningRate, f.Tfunc, f.TfuncDeriv)

	return (*MLP)(&new)
}
