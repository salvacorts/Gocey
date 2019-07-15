package common

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/salvacorts/eaopt"

	ga "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea/common"
	mn "github.com/salvacorts/go-perceptron-go/model/neural"

	"github.com/sirupsen/logrus"
)

func TestSort(t *testing.T) {
	indis := eaopt.Individuals{
		eaopt.Individual{ // A: 99 fitness, 15 neurons
			ID:      "A",
			Fitness: 99.991,
			Genome: &ga.MLP{
				NeuralLayers: []mn.NeuralLayer{ // 15 neurons
					mn.NeuralLayer{Length: 5},
					mn.NeuralLayer{Length: 10},
				},
			},
		},

		eaopt.Individual{ // B: 10 fitness, 5 neurons
			ID:      "B",
			Fitness: 10.0,
			Genome: &ga.MLP{
				NeuralLayers: []mn.NeuralLayer{ // 15 neurons
					mn.NeuralLayer{Length: 2},
					mn.NeuralLayer{Length: 3},
				},
			},
		},

		eaopt.Individual{ // C: 99 fitness, 10 neurons
			ID:      "C",
			Fitness: 99.999,
			Genome: &ga.MLP{
				NeuralLayers: []mn.NeuralLayer{ // 10 neurons
					mn.NeuralLayer{Length: 5},
					mn.NeuralLayer{Length: 5},
				},
			},
		},
	}

	// Before: [A, B, C]

	// TODO: I think I'm overwritting the original individuals
	ga.SortByFitnessAndNeurons(indis)

	// Expected: [B, C, A]
	expected := []string{"B", "C", "A"}

	for i, in := range indis {
		if in.ID != expected[i] {
			t.Errorf("Not sorted properly. Expected (%s), got (%s)", expected[i], in.ID)
		}
	}
}

func TestTrainMLP(t *testing.T) {
	filename := "../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Errorf("Cannot open %s. Error: %s", filename, err.Error())
	}

	logrus.SetOutput(os.Stdout)
	logrus.SetLevel(logrus.ErrorLevel)
	ga.Log.SetOutput(os.Stdout)
	ga.Log.SetLevel(logrus.ErrorLevel)

	_, score, err := ga.TrainMLP(string(fileContent))
	if err != nil {
		t.Errorf("Error Training MLP: %s", err.Error())
	}

	threshold := 30.0

	if score > threshold {
		t.Errorf("Got training error (%f) obove threshold (%f)", score, threshold)
	} else {
		t.Logf("Got Error: %f\n", score)
	}
}
