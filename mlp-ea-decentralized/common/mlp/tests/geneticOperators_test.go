package mlp

import (
	"io/ioutil"
	"math/rand"
	"testing"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	"github.com/salvacorts/eaopt"
	mv "github.com/salvacorts/go-perceptron-go/validation"

	"github.com/sirupsen/logrus"
)

func TestEvaluate(t *testing.T) {
	filename := "../../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Errorf("Cannot open %s. Error: %s", filename, err.Error())
	}

	logrus.SetLevel(logrus.ErrorLevel)

	var patterns, _, mapped = utils.LoadPatternsFromCSV(string(fileContent))
	train, _ := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	// Congigure MLP
	mlp.Config = mlp.MLPConfig{
		Epochs:      1,
		Folds:       1,
		Classes:     mapped,
		TrainingSet: train,
		FactoryCfg: mlp.MLPFactoryConfig{
			InputLayers:      len(patterns[0].Features),
			OutputLayers:     len(mapped),
			MinHiddenNeurons: 2,
			MaxHiddenNeurons: 20,
			Tfunc:            mlp.TransferFunc_SIGMOIDAL,
			MaxLR:            0.3,
			MinLR:            0.01,
		},
	}

	rnd := rand.New(rand.NewSource(7))
	genome := mlp.NewRandMLP(rnd)

	originalNN := genome.(*mlp.MultiLayerNetwork)

	score, err := genome.Evaluate()
	if err != nil {
		t.Errorf("Error evaluating MLP: %s", err.Error())
	}

	if originalNN.LRate != genome.(*mlp.MultiLayerNetwork).LRate {
		t.Error("LRate after evaluation is different from original LRate")
	}

	for i, layer := range originalNN.NeuralLayers {
		if layer.Length != genome.(*mlp.MultiLayerNetwork).NeuralLayers[i].Length {
			t.Errorf("In layer %d, Length (%d) after evaluation is different from original Length (%d)",
				i, genome.(*mlp.MultiLayerNetwork).NeuralLayers[i].Length, layer.Length)
		}

		for j, neuron := range layer.NeuronUnits {
			if neuron.Bias != genome.(*mlp.MultiLayerNetwork).NeuralLayers[i].NeuronUnits[j].Bias {
				t.Errorf("Bias for neuron [%d, %d] after evaluation is different from original one", i, j)
			}

			if neuron.Delta != genome.(*mlp.MultiLayerNetwork).NeuralLayers[i].NeuronUnits[j].Delta {
				t.Errorf("Delta for neuron [%d, %d] after evaluation is different from original one", i, j)
			}

			if neuron.Lrate != genome.(*mlp.MultiLayerNetwork).NeuralLayers[i].NeuronUnits[j].Lrate {
				t.Errorf("Lrate for neuron [%d, %d] after evaluation is different from original one", i, j)
			}

			if neuron.Value != genome.(*mlp.MultiLayerNetwork).NeuralLayers[i].NeuronUnits[j].Value {
				t.Errorf("Value for neuron [%d, %d] after evaluation is different from original one", i, j)
			}

			if len(neuron.Weights) != len(genome.(*mlp.MultiLayerNetwork).NeuralLayers[i].NeuronUnits[j].Weights) {
				t.Errorf("Weights for neuron [%d, %d] after evaluation (%d) have different length from original one ()%d",
					i, j, len(neuron.Weights), len(genome.(*mlp.MultiLayerNetwork).NeuralLayers[i].NeuronUnits[j].Weights))
			}

			for k, weight := range neuron.Weights {
				if weight != genome.(*mlp.MultiLayerNetwork).NeuralLayers[i].NeuronUnits[j].Weights[k] {
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

func TestTrain(t *testing.T) {
	filename := "../../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Errorf("Cannot open %s. Error: %s", filename, err.Error())
	}

	logrus.SetLevel(logrus.ErrorLevel)

	var patterns, _, mapped = utils.LoadPatternsFromCSV(string(fileContent))
	train, _ := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	// Congigure MLP
	mlp.Config = mlp.MLPConfig{
		Epochs:      1,
		Folds:       1,
		Classes:     mapped,
		TrainingSet: train,
		FactoryCfg: mlp.MLPFactoryConfig{
			InputLayers:      len(patterns[0].Features),
			OutputLayers:     len(mapped),
			MinHiddenNeurons: 2,
			MaxHiddenNeurons: 20,
			Tfunc:            mlp.TransferFunc_SIGMOIDAL,
			MaxLR:            0.3,
			MinLR:            0.01,
		},
	}

	rnd := rand.New(rand.NewSource(7))
	originalNN := mlp.NewRandMLP(rnd)

	score, err := originalNN.Evaluate()
	if err != nil {
		t.Errorf("Error evaluating MLP: %s", err.Error())
	}

	new := mlp.Train(originalNN, rnd)

	newScore, err := new.Evaluate()
	if err != nil {
		t.Errorf("Error evaluating new MLP: %s", err.Error())
	}

	if newScore > score {
		t.Errorf("Score after Training (%f) is not lower than before (%f",
			newScore, score)
	} else {
		t.Logf("Before Train: %f - Now: %f", score, newScore)
	}

}

func TestMutate(t *testing.T) {
	logrus.SetLevel(logrus.ErrorLevel)

	nn := mlp.MultiLayerNetwork{}
	rgn := rand.New(rand.NewSource(7))
	size := 5
	nWeights := 3

	nn.LRate = 0.5
	nn.NeuralLayers = make([]mlp.NeuralLayer, 3)
	nn.NeuralLayers[1].Length = int64(size)
	nn.NeuralLayers[1].NeuronUnits = make([]mlp.NeuronUnit, size)
	nn.NeuralLayers[2].Length = 2
	nn.NeuralLayers[2].NeuronUnits = make([]mlp.NeuronUnit, 2)

	for i := 0; i < size; i++ {
		nn.NeuralLayers[1].NeuronUnits[i].Weights = make([]float64, nWeights)

		for j := 0; j < nWeights; j++ {
			nn.NeuralLayers[1].NeuronUnits[i].Weights[j] =
				0.5
		}
	}

	original := nn.Clone().(*mlp.MultiLayerNetwork)

	mlp.Config.MutateRate = 1
	nn.Mutate(rgn)

	equals := true

	// Check if weights have mutated out of the range [-0.1, 0.1]
	for i, layer := range nn.NeuralLayers {
		for j, neuron := range layer.NeuronUnits {
			for k, weight := range neuron.Weights {

				if weight > original.NeuralLayers[i].NeuronUnits[j].Weights[k]+0.1 ||
					weight < original.NeuralLayers[i].NeuronUnits[j].Weights[k]-0.1 {
					t.Errorf("In neuron [%d, %d], weight mutated out of range [-0.1, 0.1]", i, j)
				}

				if weight != original.NeuralLayers[i].NeuronUnits[j].Weights[k] {
					equals = false
				}
			}
		}
	}

	if equals {
		t.Errorf("Mutation had no effect")
	}

	// Check if learning rate has mutated out of range [-0.05, 0.05]
	if nn.LRate > original.LRate+0.05 ||
		nn.LRate < original.LRate-0.05 {
		t.Errorf("L_rate has mutated out of range [-0.05, 0.05]")
	}

	// Check if learning rate is out of the range (0, 1]
	if nn.LRate > 1 || nn.LRate <= 0 {
		t.Errorf("L_rate is out of range (0, 1]")
	}
}

func TestAddNeuron(t *testing.T) {
	logrus.SetLevel(logrus.ErrorLevel)

	nn := mlp.MultiLayerNetwork{}
	rgn := rand.New(rand.NewSource(7))
	size := 5

	nn.NeuralLayers = make([]mlp.NeuralLayer, 3)
	nn.NeuralLayers[1].Length = int64(size)
	nn.NeuralLayers[1].NeuronUnits = make([]mlp.NeuronUnit, size)
	nn.NeuralLayers[2].Length = 2
	nn.NeuralLayers[2].NeuronUnits = make([]mlp.NeuronUnit, 2)

	for i := 0; i < 2; i++ {
		nn.NeuralLayers[2].NeuronUnits[i].Weights = make([]float64, size)
	}

	mlp.AddNeuron(&nn, rgn)

	if nn.NeuralLayers[1].Length != int64(size+1) {
		t.Errorf("Length attr does not match to expected. Got (%d), expected (%d)",
			nn.NeuralLayers[1].Length, size+1)
	}

	if len(nn.NeuralLayers[1].NeuronUnits) != size+1 {
		t.Errorf("Slice length does not match to expected. Got (%d), expected (%d)",
			len(nn.NeuralLayers[1].NeuronUnits), size+1)
	}

	for i, neuron := range nn.NeuralLayers[2].NeuronUnits {
		if len(neuron.Weights) != size+1 {
			t.Errorf("At neuron [%d, %d], missing weights for added neuron", 2, i)
		}
	}
}

func TestRemoveNeuron(t *testing.T) {
	logrus.SetLevel(logrus.ErrorLevel)

	nn := mlp.MultiLayerNetwork{}
	rgn := rand.New(rand.NewSource(7))
	size := 5

	nn.NeuralLayers = make([]mlp.NeuralLayer, 3)
	nn.NeuralLayers[1].Length = int64(size)
	nn.NeuralLayers[1].NeuronUnits = make([]mlp.NeuronUnit, size)
	nn.NeuralLayers[2].Length = 2
	nn.NeuralLayers[2].NeuronUnits = make([]mlp.NeuronUnit, 2)

	for i := 0; i < 2; i++ {
		nn.NeuralLayers[2].NeuronUnits[i].Weights = make([]float64, size)
	}

	mlp.RemoveNeuron(&nn, rgn)

	if nn.NeuralLayers[1].Length != int64(size-1) {
		t.Errorf("Length attr does not match to expected. Got (%d), expected (%d)",
			nn.NeuralLayers[1].Length, size-1)
	}

	if len(nn.NeuralLayers[1].NeuronUnits) != size-1 {
		t.Errorf("Slice length does not match to expected. Got (%d), expected (%d)",
			len(nn.NeuralLayers[1].NeuronUnits), size-1)
	}

	for i, neuron := range nn.NeuralLayers[2].NeuronUnits {
		if len(neuron.Weights) != size-1 {
			t.Errorf("At neuron [%d, %d], more weights than neurons in previous layer", 1, i)
		}
	}
}

func TestClone(t *testing.T) {
	logrus.SetLevel(logrus.ErrorLevel)

	nn := mlp.MultiLayerNetwork{}
	size := 5

	nn.NeuralLayers = make([]mlp.NeuralLayer, 3)
	nn.NeuralLayers[1].Length = int64(size)
	nn.NeuralLayers[1].NeuronUnits = make([]mlp.NeuronUnit, size)
	nn.NeuralLayers[2].Length = 2
	nn.NeuralLayers[2].NeuronUnits = make([]mlp.NeuronUnit, 2)

	for i := 0; i < 2; i++ {
		nn.NeuralLayers[2].NeuronUnits[i].Weights =
			[]float64{1.0, 2.0, 3.0}
	}

	new := nn.Clone().(*mlp.MultiLayerNetwork)

	new.NeuralLayers[2].Length = -7

	if new.NeuralLayers[2].Length == nn.NeuralLayers[2].Length {
		t.Error("Deep copy failed, modifying a neural layer modifies the original one")
	}

	new.NeuralLayers[2].NeuronUnits[0].Bias = -7.0

	if new.NeuralLayers[2].NeuronUnits[0].Bias == nn.NeuralLayers[2].NeuronUnits[0].Bias {
		t.Error("Deep copy failed, modifying a neuron unit modifies the original one")
	}

	new.NeuralLayers[2].NeuronUnits[0].Weights[1] = -7.0

	if new.NeuralLayers[2].NeuronUnits[0].Weights[1] == nn.NeuralLayers[2].NeuronUnits[0].Weights[1] {
		t.Error("Deep copy failed, modifying a weight modifies the original one")
	}
}

func TestCrossover(t *testing.T) {
	logrus.SetLevel(logrus.ErrorLevel)
	rng := rand.New(rand.NewSource(2153681235))
	// It will swap indexes 1 andf 3

	size1 := 7
	nn1 := mlp.MultiLayerNetwork{}
	nn1.NeuralLayers = make([]mlp.NeuralLayer, 3)
	nn1.NeuralLayers[1].Length = int64(size1)
	nn1.NeuralLayers[1].NeuronUnits = make([]mlp.NeuronUnit, size1)
	nn1.NeuralLayers[2].Length = 2
	nn1.NeuralLayers[2].NeuronUnits = make([]mlp.NeuronUnit, 2)
	for i := 0; i < size1; i++ {
		nn1.NeuralLayers[1].NeuronUnits[i].Weights =
			[]float64{0}
	}

	size2 := 10
	nn2 := mlp.MultiLayerNetwork{}
	nn2.NeuralLayers = make([]mlp.NeuralLayer, 3)
	nn2.NeuralLayers[1].Length = int64(size2)
	nn2.NeuralLayers[1].NeuronUnits = make([]mlp.NeuronUnit, size2)
	nn2.NeuralLayers[2].Length = 2
	nn2.NeuralLayers[2].NeuronUnits = make([]mlp.NeuronUnit, 2)
	for i := 0; i < size2; i++ {
		nn2.NeuralLayers[1].NeuronUnits[i].Weights =
			[]float64{1}
	}

	o1, o2 := nn1.Clone(), nn2.Clone()

	o1.Crossover(o2, rng)

	if o2.(*mlp.MultiLayerNetwork).NeuralLayers[1].NeuronUnits[1].Weights[0] !=
		nn1.NeuralLayers[1].NeuronUnits[1].Weights[0] {
		t.Errorf("Failed to create first offspring")
	}

	if o1.(*mlp.MultiLayerNetwork).NeuralLayers[1].NeuronUnits[1].Weights[0] !=
		nn2.NeuralLayers[1].NeuronUnits[1].Weights[0] {
		t.Errorf("Failed to create second offspring")
	}
}

func TestSort(t *testing.T) {
	indis := []eaopt.Individual{
		eaopt.Individual{ // A: 99 fitness, 15 neurons
			ID:      "A",
			Fitness: 99.991,
			Genome: &mlp.MultiLayerNetwork{
				NeuralLayers: []mlp.NeuralLayer{ // 15 neurons
					mlp.NeuralLayer{Length: 5},
					mlp.NeuralLayer{Length: 10},
				},
			},
		},

		eaopt.Individual{ // B: 10 fitness, 5 neurons
			ID:      "B",
			Fitness: 10.0,
			Genome: &mlp.MultiLayerNetwork{
				NeuralLayers: []mlp.NeuralLayer{ // 15 neurons
					mlp.NeuralLayer{Length: 2},
					mlp.NeuralLayer{Length: 3},
				},
			},
		},

		eaopt.Individual{ // C: 99 fitness, 10 neurons
			ID:      "C",
			Fitness: 99.999,
			Genome: &mlp.MultiLayerNetwork{
				NeuralLayers: []mlp.NeuralLayer{ // 10 neurons
					mlp.NeuralLayer{Length: 5},
					mlp.NeuralLayer{Length: 5},
				},
			},
		},
	}

	// Before: [A, B, C]

	// I'm overwritting the original individuals
	indis = mlp.SortByFitnessAndNeurons(indis, 100)

	// Expected: [B, C, A]
	expected := []string{"B", "C", "A"}

	for i, in := range indis {
		if in.ID != expected[i] {
			t.Errorf("Not sorted properly. Expected (%s), got (%s)", expected[i], in.ID)
		}
	}
}
