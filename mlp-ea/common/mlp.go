package common

import (
	"math/rand"

	utils "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	"github.com/salvacorts/eaopt"
	mn "github.com/salvacorts/go-perceptron-go/model/neural"
	mv "github.com/salvacorts/go-perceptron-go/validation"
	"github.com/sirupsen/logrus"
)

// MLP is a type of MultiLayerNetwork that implements the eaopt.Genome interface
type MLP mn.MultiLayerNetwork

// NewRandMLP creates a randomly initialized MLP
func NewRandMLP(rng *rand.Rand) eaopt.Genome {
	layers := []int{
		Config.FactoryCfg.InputLayers,
		utils.RandIntInRange(
			Config.FactoryCfg.MinHiddenNeurons,
			Config.FactoryCfg.MaxHiddenNeurons,
			rng),
		Config.FactoryCfg.OutputLayers,
	}

	learningRate := Config.FactoryCfg.MaxLR +
		utils.RandFloatInRange(Config.FactoryCfg.MinLR, Config.FactoryCfg.MaxLR, rng)

	new := mn.PrepareMLPNet(
		layers, learningRate, Config.FactoryCfg.Tfunc, Config.FactoryCfg.TfuncDeriv)

	return (*MLP)(&new)
}

// Evaluate a MLP by getting its accuracy
// TODO: Compare also number of neurons
func (mlp *MLP) Evaluate() (float64, error) {
	copy := mlp.Clone().(*MLP)

	scores := mv.MLPKFoldValidation(
		(*mn.MultiLayerNetwork)(copy), *Config.TrainingSet, Config.Epochs, Config.Folds, 1, *Config.Classes)

	mean := 0.0
	for _, s := range scores {
		mean += s
	}
	mean /= float64(len(scores))

	return 100 - mean, nil
}

// Mutate modifies the weights of certain neurons, at random, depending on the application rate.
// dding or subtracting a small random number that follows uniform distribution with the interval [-0.1, 0.1].
// The learning rate is modified by adding a small random number that follows uniform distribution
// in the interval [-0.05, 0.05]
func (mlp *MLP) Mutate(rng *rand.Rand) {
	indexNL := utils.RandIntInRange(1, len(mlp.NeuralLayers)-1, rng)

	// Modifies the weights of certain neurons, at random, depending on the application rate
	for i := range mlp.NeuralLayers[indexNL].NeuronUnits {
		if rng.Float64() < MutateRate {
			u := utils.RandFloatInRange(-0.1, 0.1, rng) * ScalingFactor

			for j := range mlp.NeuralLayers[indexNL].NeuronUnits[i].Weights {
				mlp.NeuralLayers[indexNL].NeuronUnits[i].Weights[j] += u
			}
		}
	}

	// Modify learning rate
	if rng.Float64() < MutateRate {
		u := utils.RandFloatInRange(-0.05, 0.05, rng)

		newLR := mlp.L_rate + u

		// If by adding u the new LR is too high or small,
		// substract it to be in the desired range
		if newLR <= 0 || newLR > 1 {
			newLR = mlp.L_rate - u
		}

		mlp.L_rate = newLR
	}

	Log.WithFields(logrus.Fields{
		"level":  "debug",
		"place":  "Genetic Operator",
		"method": "Mutate",
	}).Debug("Mutate operator completed.")
}

// Crossover carries out the multipoint cross-over between two chromosome nets,
// so that two networks are obtained whose hidden layer neurons are a mixture of the
// hidden layer neurons of both parents:
// some hidden neurons along with their in and out connections, from each parent
// make one offspring and the remaining hidden neurons make the other one.
// The learningrate is swapped between the two nets.
func (mlp *MLP) Crossover(Y eaopt.Genome, rng *rand.Rand) {
	n := len(mlp.NeuralLayers)

	if len(Y.(*MLP).NeuralLayers) > n {
		n = len(Y.(*MLP).NeuralLayers)
	}

	// Perform a multipoint cross-over on each hidden layer
	for i := 1; i < n-1; i++ {
		p1 := neurons(mlp.NeuralLayers[i].NeuronUnits)
		p2 := neurons(Y.(*MLP).NeuralLayers[i].NeuronUnits)

		eaopt.CrossGNX(p1, p2, 2, rng)
	}

	// Swap learning rates
	tmp := mlp.L_rate
	mlp.L_rate = Y.(*MLP).L_rate
	Y.(*MLP).L_rate = tmp

	Log.WithFields(logrus.Fields{
		"level":  "debug",
		"place":  "Genetic Operator",
		"method": "Crossover",
	}).Debug("Crossover operator completed.")
}

// AddNeuron is intended to perform incremental learning: it starts with a small structure
// and increments it, if neccesary, by adding new hidden units
// TODO: 2 veces probabilidad de eliminar y poner limite de tamaño
func AddNeuron(in eaopt.Genome, rng *rand.Rand) eaopt.Genome {
	out := in.(*MLP)

	indexNL := utils.RandIntInRange(1, len(out.NeuralLayers)-1, rng)
	newNeuron := mn.NeuronUnit{}

	opLogger := Log.WithFields(logrus.Fields{
		"level":   "debug",
		"place":   "Genetic Operator",
		"method":  "AddNeuron",
		"Layer":   indexNL,
		"Size":    out.NeuralLayers[indexNL].Length,
		"MaxSize": Config.FactoryCfg.MaxHiddenNeurons,
	})

	// If there as much neurons as the maximum, return
	if out.NeuralLayers[indexNL].Length >= Config.FactoryCfg.MaxHiddenNeurons {
		opLogger.Debug("Cannot add neuron: Max neurons reached")
		return out
	}

	mn.RandomNeuronInit(&newNeuron, out.NeuralLayers[indexNL-1].Length)

	out.NeuralLayers[indexNL].Length++
	out.NeuralLayers[indexNL].NeuronUnits =
		append(out.NeuralLayers[indexNL].NeuronUnits, newNeuron)

	// Add weights for the added neuron in the following layer
	for i := 0; i < out.NeuralLayers[indexNL+1].Length; i++ {
		out.NeuralLayers[indexNL+1].NeuronUnits[i].Weights =
			append(out.NeuralLayers[indexNL+1].NeuronUnits[i].Weights,
				rng.NormFloat64()*ScalingFactor)
	}

	opLogger.Debugf("AddNeuron operator completed. Now %d",
		out.NeuralLayers[indexNL].Length)

	return out
}

// RemoveNeuron eliminates one hidden neuron at random
func RemoveNeuron(in eaopt.Genome, rng *rand.Rand) eaopt.Genome {
	out := in.(*MLP)

	indexNL := utils.RandIntInRange(1, len(out.NeuralLayers)-1, rng)
	indexNU := utils.RandIntInRange(0, out.NeuralLayers[indexNL].Length, rng)

	opLogger := Log.WithFields(logrus.Fields{
		"level":   "debug",
		"place":   "Genetic Operator",
		"method":  "RemoveNeuron",
		"Layer":   indexNL,
		"Size":    out.NeuralLayers[indexNL].Length,
		"MinSize": Config.FactoryCfg.MinHiddenNeurons,
		"Neuron":  indexNU,
	})

	// If there are just the minimum of neurons, do nothing
	if out.NeuralLayers[indexNL].Length <= Config.FactoryCfg.MinHiddenNeurons {
		opLogger.Debug("Cannot remove neuron: Min neurons reached")
		return out
	}

	out.NeuralLayers[indexNL].Length--
	out.NeuralLayers[indexNL].NeuronUnits =
		utils.RemoveNeuron(out.NeuralLayers[indexNL].NeuronUnits, indexNU)

	// Remove weights for the removed neuron in the following layer
	for i := 0; i < out.NeuralLayers[indexNL+1].Length; i++ {
		out.NeuralLayers[indexNL+1].NeuronUnits[i].Weights =
			utils.RemoveF64(out.NeuralLayers[indexNL+1].NeuronUnits[i].Weights, indexNU)
	}

	opLogger.Debugf("RemoveNeuron operator completed. Now %d",
		out.NeuralLayers[indexNL].Length)

	return out
}

// SubstituteNeuron replaces one hiddenlayer neuron at random with a new one,
// initialized with random weights
func SubstituteNeuron(in eaopt.Genome, rng *rand.Rand) eaopt.Genome {
	out := in.(*MLP)

	indexNL := utils.RandIntInRange(1, len(out.NeuralLayers)-1, rng)
	indexNU := utils.RandIntInRange(0, len(out.NeuralLayers[indexNL].NeuronUnits), rng)
	dim := len(out.NeuralLayers[indexNL].NeuronUnits[indexNU].Weights)

	for i := 0; i < dim; i++ {
		out.NeuralLayers[indexNL].NeuronUnits[indexNU].Weights[i] =
			rng.NormFloat64() * ScalingFactor
	}

	Log.WithFields(logrus.Fields{
		"level":  "debug",
		"place":  "Genetic Operator",
		"method": "SubstituteNeuron",
		"Layer":  indexNL,
		"Neuron": indexNU,
	}).Debug("SubstituteNeuron operator completed.")

	return out
}

// Train is used to train the individual-net for a certain number of generations, using BP algorithm.
func Train(in eaopt.Genome, rng *rand.Rand) eaopt.Genome {
	out := in.(*MLP)

	mn.MLPTrain(
		(*mn.MultiLayerNetwork)(out),
		*Config.TrainingSet,
		*Config.Classes, TrainEpochs,
		true)

	Log.WithFields(logrus.Fields{
		"level":  "debug",
		"place":  "Genetic Operator",
		"method": "Train",
		"epochs": Config.Epochs,
	}).Debug("Train operator completed.")

	return out
}

// TODO: AddLayer and RemoveLayer operators

// Clone a MLP to produce a new one that points to a different one by doing a deep copy.
func (mlp *MLP) Clone() eaopt.Genome {
	new := MLP{
		L_rate:       mlp.L_rate,
		T_func:       mlp.T_func,
		T_func_d:     mlp.T_func_d,
		NeuralLayers: make([]mn.NeuralLayer, len(mlp.NeuralLayers)),
	}

	copy(new.NeuralLayers, mlp.NeuralLayers)

	for i := range new.NeuralLayers {
		new.NeuralLayers[i].Length = mlp.NeuralLayers[i].Length

		new.NeuralLayers[i].NeuronUnits =
			neurons(mlp.NeuralLayers[i].NeuronUnits).Copy().(neurons)
	}

	return &new
}
