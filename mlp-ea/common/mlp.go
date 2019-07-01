package common

import (
	"math/rand"

	eaopt "github.com/MaxHalford/eaopt"
	mn "github.com/made2591/go-perceptron-go/model/neural"
	v "github.com/made2591/go-perceptron-go/validation"
)

// MLPConfig Stores the common variables for training and evaluating all MLPs generated by the EA
type MLPConfig struct {
	Epochs    int
	Classes   *[]string
	TrainData *[]mn.Pattern
	TestData  *[]mn.Pattern
}

// MLP is a type of MultiLayerNetwork that implements Genome interface
type MLP struct {
	nn     mn.MultiLayerNetwork
	config MLPConfig
}

// Evaluate a MLP by getting its accuracy
func (mlp MLP) Evaluate() (float64, error) {
	// Train the network
	mn.MLPTrain(&mlp.nn, *mlp.config.TrainData, *mlp.config.Classes, mlp.config.epochs)

	// Get nn accuracy
	mean, _ := v.RNNValidation(&mlp.nn, *mlp.config.TestData, mlp.config.epochs, 0)

	return mean, nil
}

// Mutate a Vector by resampling each element from a normal distribution with
// probability 0.8.
func (mlp MLP) Mutate(rng *rand.Rand) {
	// TODO: Mutate
}

// Crossover a Vector with another Vector by applying uniform crossover.
func (mlp MLP) Crossover(Y eaopt.Genome, rng *rand.Rand) {
	// TODO: Cross
}

// Clone a MLP to produce a new one that points to a different one.
func (mlp MLP) Clone() eaopt.Genome {
	return MLP{
		nn:     mlp.nn,
		config: mlp.config,
	}
}