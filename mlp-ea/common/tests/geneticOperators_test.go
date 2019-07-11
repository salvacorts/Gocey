package common

import (
	"io/ioutil"
	"math/rand"
	"testing"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea/common"
	ga "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea/common"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	mn "github.com/salvacorts/go-perceptron-go/model/neural"
	v "github.com/salvacorts/go-perceptron-go/validation"
	mlpLogger "github.com/sirupsen/logrus"
)

func TestEvaluate(t *testing.T) {
	filename := "../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Errorf("Cannot open %s. Error: %s", filename, err.Error())
	}

	mlpLogger.SetLevel(mlpLogger.ErrorLevel)

	var patterns, _, mapped = utils.LoadPatternsFromCSV(string(fileContent))
	train, validation := v.TrainTestPatternSplit(patterns, 0.8, 1)

	// Congigure MLP Factory
	mlpFactory := common.MLPFactory{
		InputLayers:     len(patterns[0].Features),
		OutputLayers:    len(mapped),
		MaxHiddenLayers: 30,
		Tfunc:           mn.SigmoidalTransfer,
		TfuncDeriv:      mn.SigmoidalTransferDerivate,
		MaxLR:           0.3,
		MinLR:           0.01,

		Config: common.MLPConfig{
			Epochs:        1,
			Folds:         1,
			Classes:       &mapped,
			TrainingSet:   &train,
			ValidationSet: &validation,
		},
	}

	rnd := rand.New(rand.NewSource(7))
	genome := mlpFactory.NewRandMLP(rnd)

	originalNN := genome.(*ga.MLP).NeuralNet

	score, err := genome.Evaluate()
	if err != nil {
		t.Errorf("Error training MLP: %s", err.Error())
	}

	if originalNN.L_rate != genome.(*ga.MLP).NeuralNet.L_rate {
		t.Error("L_rate after evaluation is different from original L_rate")
	}

	for i, layer := range originalNN.NeuralLayers {
		if layer.Length != genome.(*ga.MLP).NeuralNet.NeuralLayers[i].Length {
			t.Errorf("In layer %d, Length (%d) after evaluation is different from original Length (%d)",
				i, genome.(*ga.MLP).NeuralNet.NeuralLayers[i].Length, layer.Length)
		}

		for j, neuron := range layer.NeuronUnits {
			if neuron.Bias != genome.(*ga.MLP).NeuralNet.NeuralLayers[i].NeuronUnits[j].Bias {
				t.Errorf("Bias for neuron [%d, %d] after evaluation is different from original one", i, j)
			}

			if neuron.Delta != genome.(*ga.MLP).NeuralNet.NeuralLayers[i].NeuronUnits[j].Delta {
				t.Errorf("Delta for neuron [%d, %d] after evaluation is different from original one", i, j)
			}

			if neuron.Lrate != genome.(*ga.MLP).NeuralNet.NeuralLayers[i].NeuronUnits[j].Lrate {
				t.Errorf("Lrate for neuron [%d, %d] after evaluation is different from original one", i, j)
			}

			if neuron.Value != genome.(*ga.MLP).NeuralNet.NeuralLayers[i].NeuronUnits[j].Value {
				t.Errorf("Value for neuron [%d, %d] after evaluation is different from original one", i, j)
			}

			if len(neuron.Weights) != len(genome.(*ga.MLP).NeuralNet.NeuralLayers[i].NeuronUnits[j].Weights) {
				t.Errorf("Weights for neuron [%d, %d] after evaluation (%d) have different length from original one ()%d",
					i, j, len(neuron.Weights), len(genome.(*ga.MLP).NeuralNet.NeuralLayers[i].NeuronUnits[j].Weights))
			}

			for k, weight := range neuron.Weights {
				if weight != genome.(*ga.MLP).NeuralNet.NeuralLayers[i].NeuronUnits[j].Weights[k] {
					t.Errorf("Weight %d at neuron [%d, %d] after evaluation is different from original one", k, i, j)
				}
			}
		}
	}

	expectedScore := 0.0
	if score < expectedScore {
		t.Errorf("Got training score (%f) under threshold (%f)", score, expectedScore)
	}
}

func TestMutate(t *testing.T) {
	mlpLogger.SetLevel(mlpLogger.ErrorLevel)

	mlp := common.MLP{}
	rgn := rand.New(rand.NewSource(7))
	size := 5
	nWeights := 3

	mlp.NeuralNet.L_rate = 0.5
	mlp.NeuralNet.NeuralLayers = make([]mn.NeuralLayer, 3)
	mlp.NeuralNet.NeuralLayers[1].Length = size
	mlp.NeuralNet.NeuralLayers[1].NeuronUnits = make([]mn.NeuronUnit, size)
	mlp.NeuralNet.NeuralLayers[2].Length = 2
	mlp.NeuralNet.NeuralLayers[2].NeuronUnits = make([]mn.NeuronUnit, 2)

	for i := 0; i < size; i++ {
		mlp.NeuralNet.NeuralLayers[1].NeuronUnits[i].Weights = make([]float64, nWeights)

		for j := 0; j < nWeights; j++ {
			mlp.NeuralNet.NeuralLayers[1].NeuronUnits[i].Weights[j] =
				0.5
		}
	}

	original := mlp.Clone().(*common.MLP)

	mlp.Mutate(rgn)

	equals := true

	// Check if weights have mutated aout of the range [-0.1, 0.1]
	for i, layer := range mlp.NeuralNet.NeuralLayers {
		for j, neuron := range layer.NeuronUnits {
			for k, weight := range neuron.Weights {

				if weight > original.NeuralNet.NeuralLayers[i].NeuronUnits[j].Weights[k]+0.1 ||
					weight < original.NeuralNet.NeuralLayers[i].NeuronUnits[j].Weights[k]-0.1 {
					t.Errorf("In neuron [%d, %d], weight mutated out of range [-0.1, 0.1]", i, j)
				}

				if weight != original.NeuralNet.NeuralLayers[i].NeuronUnits[j].Weights[k] {
					equals = false
				}
			}
		}
	}

	if equals {
		t.Errorf("Mutation had no effect")
	}

	// Check if learning rate has mutated out of range [-0.05, 0.05]
	if mlp.NeuralNet.L_rate > original.NeuralNet.L_rate+0.05 ||
		mlp.NeuralNet.L_rate < original.NeuralNet.L_rate-0.05 {
		t.Errorf("L_rate has mutated out of range [-0.05, 0.05]")
	}

	// Check if learning rate is out of the range (0, 1]
	if mlp.NeuralNet.L_rate > 1 || mlp.NeuralNet.L_rate <= 0 {
		t.Errorf("L_rate is out of range (0, 1]")
	}
}

func TestAddNeuron(t *testing.T) {
	mlpLogger.SetLevel(mlpLogger.ErrorLevel)

	mlp := common.MLP{}
	rgn := rand.New(rand.NewSource(7))
	size := 5

	mlp.NeuralNet.NeuralLayers = make([]mn.NeuralLayer, 3)
	mlp.NeuralNet.NeuralLayers[1].Length = size
	mlp.NeuralNet.NeuralLayers[1].NeuronUnits = make([]mn.NeuronUnit, size)
	mlp.NeuralNet.NeuralLayers[2].Length = 2
	mlp.NeuralNet.NeuralLayers[2].NeuronUnits = make([]mn.NeuronUnit, 2)

	for i := 0; i < 2; i++ {
		mlp.NeuralNet.NeuralLayers[2].NeuronUnits[i].Weights = make([]float64, size)
	}

	common.AddNeuron(&mlp, rgn)

	if mlp.NeuralNet.NeuralLayers[1].Length != size+1 {
		t.Errorf("Length attr does not match to expected. Got (%d), expected (%d)",
			mlp.NeuralNet.NeuralLayers[1].Length, size+1)
	}

	if len(mlp.NeuralNet.NeuralLayers[1].NeuronUnits) != size+1 {
		t.Errorf("Slice length does not match to expected. Got (%d), expected (%d)",
			len(mlp.NeuralNet.NeuralLayers[1].NeuronUnits), size+1)
	}

	for i, neuron := range mlp.NeuralNet.NeuralLayers[2].NeuronUnits {
		if len(neuron.Weights) != size+1 {
			t.Errorf("At neuron [%d, %d], missing weights for added neuron", 1, i)
		}
	}
}

func TestRemoveNeuron(t *testing.T) {
	mlpLogger.SetLevel(mlpLogger.ErrorLevel)

	mlp := common.MLP{}
	rgn := rand.New(rand.NewSource(7))
	size := 5

	mlp.NeuralNet.NeuralLayers = make([]mn.NeuralLayer, 3)
	mlp.NeuralNet.NeuralLayers[1].Length = size
	mlp.NeuralNet.NeuralLayers[1].NeuronUnits = make([]mn.NeuronUnit, size)
	mlp.NeuralNet.NeuralLayers[2].Length = 2
	mlp.NeuralNet.NeuralLayers[2].NeuronUnits = make([]mn.NeuronUnit, 2)

	for i := 0; i < 2; i++ {
		mlp.NeuralNet.NeuralLayers[2].NeuronUnits[i].Weights = make([]float64, size)
	}

	common.RemoveNeuron(&mlp, rgn)

	if mlp.NeuralNet.NeuralLayers[1].Length != size-1 {
		t.Errorf("Length attr does not match to expected. Got (%d), expected (%d)",
			mlp.NeuralNet.NeuralLayers[1].Length, size-1)
	}

	if len(mlp.NeuralNet.NeuralLayers[1].NeuronUnits) != size-1 {
		t.Errorf("Slice length does not match to expected. Got (%d), expected (%d)",
			len(mlp.NeuralNet.NeuralLayers[1].NeuronUnits), size-1)
	}

	for i, neuron := range mlp.NeuralNet.NeuralLayers[2].NeuronUnits {
		if len(neuron.Weights) != size-1 {
			t.Errorf("At neuron [%d, %d], more weights than neurons in previous layer", 1, i)
		}
	}
}

func TestClone(t *testing.T) {
	mlpLogger.SetLevel(mlpLogger.ErrorLevel)

	mlp := common.MLP{}
	size := 5

	mlp.NeuralNet.NeuralLayers = make([]mn.NeuralLayer, 3)
	mlp.NeuralNet.NeuralLayers[1].Length = size
	mlp.NeuralNet.NeuralLayers[1].NeuronUnits = make([]mn.NeuronUnit, size)
	mlp.NeuralNet.NeuralLayers[2].Length = 2
	mlp.NeuralNet.NeuralLayers[2].NeuronUnits = make([]mn.NeuronUnit, 2)

	for i := 0; i < 2; i++ {
		mlp.NeuralNet.NeuralLayers[2].NeuronUnits[i].Weights =
			[]float64{1.0, 2.0, 3.0}
	}

	new := mlp.Clone().(*common.MLP)

	new.NeuralNet.NeuralLayers[2].Length = -7

	if new.NeuralNet.NeuralLayers[2].Length == mlp.NeuralNet.NeuralLayers[2].Length {
		t.Error("Deep copy failed, modifying a neural layer modifies the original one")
	}

	new.NeuralNet.NeuralLayers[2].NeuronUnits[0].Bias = -7.0

	if new.NeuralNet.NeuralLayers[2].NeuronUnits[0].Bias == mlp.NeuralNet.NeuralLayers[2].NeuronUnits[0].Bias {
		t.Error("Deep copy failed, modifying a neuron unit modifies the original one")
	}

	new.NeuralNet.NeuralLayers[2].NeuronUnits[0].Weights[1] = -7.0

	if new.NeuralNet.NeuralLayers[2].NeuronUnits[0].Weights[1] == mlp.NeuralNet.NeuralLayers[2].NeuronUnits[0].Weights[1] {
		t.Error("Deep copy failed, modifying a weight modifies the original one")
	}
}

func TestCrossover(t *testing.T) {
	mlpLogger.SetLevel(mlpLogger.ErrorLevel)
	rng := rand.New(rand.NewSource(2153681235))
	// It will swap indexes 1 andf 3

	size1 := 7
	mlp1 := common.MLP{}
	mlp1.NeuralNet.NeuralLayers = make([]mn.NeuralLayer, 3)
	mlp1.NeuralNet.NeuralLayers[1].Length = size1
	mlp1.NeuralNet.NeuralLayers[1].NeuronUnits = make([]mn.NeuronUnit, size1)
	mlp1.NeuralNet.NeuralLayers[2].Length = 2
	mlp1.NeuralNet.NeuralLayers[2].NeuronUnits = make([]mn.NeuronUnit, 2)
	for i := 0; i < size1; i++ {
		mlp1.NeuralNet.NeuralLayers[1].NeuronUnits[i].Weights =
			[]float64{0}
	}

	size2 := 10
	mlp2 := common.MLP{}
	mlp2.NeuralNet.NeuralLayers = make([]mn.NeuralLayer, 3)
	mlp2.NeuralNet.NeuralLayers[1].Length = size2
	mlp2.NeuralNet.NeuralLayers[1].NeuronUnits = make([]mn.NeuronUnit, size2)
	mlp2.NeuralNet.NeuralLayers[2].Length = 2
	mlp2.NeuralNet.NeuralLayers[2].NeuronUnits = make([]mn.NeuronUnit, 2)
	for i := 0; i < size2; i++ {
		mlp2.NeuralNet.NeuralLayers[1].NeuronUnits[i].Weights =
			[]float64{1}
	}

	o1, o2 := mlp1.Clone(), mlp2.Clone()

	o1.Crossover(o2, rng)

	if o2.(*common.MLP).NeuralNet.NeuralLayers[1].NeuronUnits[1].Weights[0] !=
		mlp1.NeuralNet.NeuralLayers[1].NeuronUnits[1].Weights[0] {
		t.Errorf("Failed to create first offspring")
	}

	if o1.(*common.MLP).NeuralNet.NeuralLayers[1].NeuronUnits[1].Weights[0] !=
		mlp2.NeuralNet.NeuralLayers[1].NeuronUnits[1].Weights[0] {
		t.Errorf("Failed to create second offspring")
	}
}
