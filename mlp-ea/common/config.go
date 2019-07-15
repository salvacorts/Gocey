package common

import mn "github.com/salvacorts/go-perceptron-go/model/neural"

const (
	// Precission states the fitness precission.
	// e.g. 100 is 2 decimals (default)
	Precission float64 = 100

	// ScalingFactor is Scaling factor for float64 generated random values
	ScalingFactor float64 = 0.0000000000001 // TODO: Check if this is really useful

	// TrainEpochs states how many epochs to train the MLP in the Train operator
	TrainEpochs int = 100

	// MutateRate states the probability of a neuron to be mutated once the Mutate operator is applied
	MutateRate float64 = 1

	generations uint = 100
	popSize     uint = 100

	mutProb              float64 = 0.3
	crossProb            float64 = 0.3
	addNeuronProb        float64 = 0.3
	removeNeuronProb     float64 = 0.15
	substituteNeuronProb float64 = 0.15
	trainProb            float64 = 0.1
)

// TransferFunction stands for a transfer function
type TransferFunction = func(float64) float64

// MLPFactoryConfig is a type to create a factory of MLP randomly initialized
type MLPFactoryConfig struct {
	// Network topology
	InputLayers      int
	OutputLayers     int
	MinHiddenNeurons int
	MaxHiddenNeurons int
	Tfunc            TransferFunction
	TfuncDeriv       TransferFunction

	// Training Info configuration
	MaxLR float64
	MinLR float64
}

// MLPConfig Stores the common variables for training and evaluating all MLPs generated by the EA
type MLPConfig struct {
	Epochs      int
	Folds       int
	Classes     *[]string
	TrainingSet *[]mn.Pattern
	FactoryCfg  MLPFactoryConfig
}
