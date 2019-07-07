package common

import (
	"math/rand"

	mn "github.com/made2591/go-perceptron-go/model/neural"
	mu "github.com/made2591/go-perceptron-go/util"
	mv "github.com/made2591/go-perceptron-go/validation"
	utils "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	"github.com/salvacorts/eaopt"
)

const (
	// SCALING_FACTOR is Scaling factor for float64 generated random values
	SCALING_FACTOR = 0.0000000000001
)

// MLPConfig Stores the common variables for training and evaluating all MLPs generated by the EA
type MLPConfig struct {
	Epochs        int
	Folds         int
	Classes       *[]string
	TrainingSet   *[]mn.Pattern
	ValidationSet *[]mn.Pattern
}

// MLP is a type of MultiLayerNetwork that implements Genome interface
type MLP struct {
	NeuralNet mn.MultiLayerNetwork
	Config    MLPConfig
}

// Evaluate a MLP by getting its accuracy
func (mlp MLP) Evaluate() (float64, error) {
	// Train the network
	mn.MLPTrain(&mlp.NeuralNet, *mlp.Config.TrainingSet, *mlp.Config.Classes, mlp.Config.Epochs)

	scores := mv.MLPKFoldValidation(
		&mlp.NeuralNet,
		*mlp.Config.TrainingSet,
		mlp.Config.Epochs,
		mlp.Config.Folds,
		0,
		*mlp.Config.Classes)

	accuracy, _ := mu.MaxInSlice(scores)

	return 100 - accuracy, nil
}

// Mutate modifies the weights of certain neurons, at random, depending on the application rate.
// Modifies the weights of the network after each epoch of network training,
// adding or subtracting a small random number that follows uniform distribution with the interval [-0.1, 0.1].
// The learning rate is modified by adding a small random number that follows uniform distribution
// in the interval [-0.05, 0.05]
func (mlp MLP) Mutate(rng *rand.Rand) {
	// TODO: Mutate
}

// Crossover carries out the multipoint cross-over between two chromosome nets,
// so that two networks are obtained whose hidden layer neurons are a mixture of the
// hidden layer neurons of both parents:
// some hidden neurons along with their in and out connections, from each parent
// make one offspring and the remaining hidden neurons make the other one.
// The learningrate is swapped between the two nets.
func (mlp MLP) Crossover(Y eaopt.Genome, rng *rand.Rand) {
	// TODO: Crossover
}

// AddNeuron s intended to performincremental learning: it starts with a small structure
// and increments it, if neccesary, by adding new hidden units
func AddNeuron(in eaopt.Genome, rng *rand.Rand) eaopt.Genome {
	out := in.(MLP)

	indexNL := utils.RandIntInRange(1, len(out.NeuralNet.NeuralLayers)-1, rng)
	newNeuron := mn.NeuronUnit{}

	mn.RandomNeuronInit(&newNeuron, out.NeuralNet.NeuralLayers[indexNL-1].Length)

	out.NeuralNet.NeuralLayers[indexNL].Length++
	out.NeuralNet.NeuralLayers[indexNL].NeuronUnits =
		append(out.NeuralNet.NeuralLayers[indexNL].NeuronUnits, newNeuron)

	return out
}

// RemoveNeuron eliminates one hidden neuron at random
func RemoveNeuron(in eaopt.Genome, rng *rand.Rand) eaopt.Genome {
	out := in.(MLP)

	indexNL := utils.RandIntInRange(1, len(out.NeuralNet.NeuralLayers)-1, rng)
	indexNU := utils.RandIntInRange(0, out.NeuralNet.NeuralLayers[indexNL].Length, rng)

	out.NeuralNet.NeuralLayers[indexNL].Length--
	out.NeuralNet.NeuralLayers[indexNL].NeuronUnits =
		utils.Remove(out.NeuralNet.NeuralLayers[indexNL].NeuronUnits, indexNU)

	return out
}

// SubstituteNeuron replaces one hiddenlayer neuron at random with a new one,
// initialized with random weights
func SubstituteNeuron(in eaopt.Genome, rng *rand.Rand) eaopt.Genome {
	out := in.(MLP)

	indexNL := utils.RandIntInRange(1, len(out.NeuralNet.NeuralLayers)-1, rng)
	indexNU := utils.RandIntInRange(0, len(out.NeuralNet.NeuralLayers[indexNL].NeuronUnits), rng)
	dim := len(out.NeuralNet.NeuralLayers[indexNL].NeuronUnits[indexNU].Weights)

	for i := 0; i < dim; i++ {
		out.NeuralNet.NeuralLayers[indexNL].NeuronUnits[indexNU].Weights[i] =
			rng.NormFloat64() * SCALING_FACTOR
	}

	return out
}

// Train is used to train the individual-net for a certain number of generations, using the QP algorithm.
func Train(in eaopt.Genome, rng *rand.Rand) (out eaopt.Genome) {
	// TODO: Train
	out = in
	return out
}

// Clone a MLP to produce a new one that points to a different one.
func (mlp MLP) Clone() eaopt.Genome {
	return MLP{
		NeuralNet: mlp.NeuralNet,
		Config:    mlp.Config,
	}
}
