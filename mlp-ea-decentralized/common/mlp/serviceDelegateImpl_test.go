package mlp

import (
	"io/ioutil"
	"strings"
	"testing"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	mv "github.com/salvacorts/go-perceptron-go/validation"
	"github.com/sirupsen/logrus"
)

func TestSerializeProblemDescription(t *testing.T) {
	filename := "../../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Errorf("Cannot open %s. Error: %s", filename, err.Error())
	}

	logrus.SetLevel(logrus.ErrorLevel)

	var patterns, _, mapped = utils.LoadPatternsFromCSV(string(fileContent))
	train, _ := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	mlp.Config.Epochs = 7
	mlp.Config.Folds = 7
	mlp.Config.TrainingSet = train
	mlp.Config.Classes = mapped

	delegate := mlp.DelegateImpl{}
	buff := delegate.SerializeProblemDescription()

	desc := mlp.MLPDescription{}
	err = desc.Unmarshal(buff)
	if err != nil {
		t.Errorf("Could not deserialize problem description. %s", err.Error())
	}

	if desc.Epochs != int64(mlp.Config.Epochs) {
		t.Errorf("Epochs do not match: %d vs %d.", desc.Epochs, mlp.Config.Epochs)
	}

	if desc.Folds != int64(mlp.Config.Folds) {
		t.Errorf("Folds do not match: %d vs %d.", desc.Folds, mlp.Config.Folds)
	}

	trainStr := utils.PatternsToCSV(train)
	if strings.Compare(desc.TrainDataset, trainStr) != 0 {
		t.Errorf("Training datasets do not match. Len: %d vs %d", len(desc.TrainDataset), len(trainStr))
	}

	if !equalString(desc.Classes, mlp.Config.Classes) {
		t.Errorf("Classes do not match:\n\tA: %v\n\tB: %v", desc.Classes, mlp.Config.Classes)
	}
}

func TestDeserializeProblemDescription(t *testing.T) {
	filename := "../../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Errorf("Cannot open %s. Error: %s", filename, err.Error())
	}

	logrus.SetLevel(logrus.ErrorLevel)

	var patterns, _, mapped = utils.LoadPatternsFromCSV(string(fileContent))
	train, _ := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	desc := mlp.MLPDescription{
		Epochs:       7,
		Folds:        7,
		TrainDataset: utils.PatternsToCSV(train),
		Classes:      mapped,
	}

	buff, err := desc.Marshal()
	if err != nil {
		t.Errorf("Could not serialize problem description. %s", err.Error())
	}

	delegate := mlp.DelegateImpl{}
	delegate.DeserializeProblemDescription(buff)

	if desc.Epochs != int64(mlp.Config.Epochs) {
		t.Errorf("Epochs do not match: %d vs %d.", desc.Epochs, mlp.Config.Epochs)
	}

	if desc.Folds != int64(mlp.Config.Folds) {
		t.Errorf("Folds do not match: %d vs %d.", desc.Folds, mlp.Config.Folds)
	}

	trainStr := utils.PatternsToCSV(train)
	if strings.Compare(desc.TrainDataset, trainStr) != 0 {
		t.Errorf("Training datasets do not match. Len: %d vs %d", len(desc.TrainDataset), len(trainStr))
	}

	if !equalString(desc.Classes, mlp.Config.Classes) {
		t.Errorf("Classes do not match:\n\tA: %v\n\tB: %v", desc.Classes, mlp.Config.Classes)
	}
}

func TestSerializeGenome(t *testing.T) {
	nn := &mlp.MultiLayerNetwork{}
	size := 5

	nn.LRate = 0.5
	nn.NeuralLayers = make([]mlp.NeuralLayer, 3)
	nn.NeuralLayers[1].Length = int64(size)
	nn.NeuralLayers[1].NeuronUnits = make([]mlp.NeuronUnit, size)
	nn.NeuralLayers[2].Length = 2
	nn.NeuralLayers[2].NeuronUnits = make([]mlp.NeuronUnit, 2)
	nn.NeuralLayers[1].NeuronUnits[1].Bias = 0.1
	nn.NeuralLayers[1].NeuronUnits[1].Value = 0.2
	nn.NeuralLayers[1].NeuronUnits[1].Lrate = 0.3
	nn.NeuralLayers[1].NeuronUnits[1].Delta = 0.4
	nn.NeuralLayers[1].NeuronUnits[1].Weights = make([]float64, 2)
	nn.NeuralLayers[1].NeuronUnits[1].Weights[0] = 0.01
	nn.NeuralLayers[1].NeuronUnits[1].Weights[1] = 0.02

	delegate := mlp.DelegateImpl{}
	buff := delegate.SerializeGenome(nn)

	nn2 := mlp.MultiLayerNetwork{}
	err := nn2.Unmarshal(buff)
	if err != nil {
		t.Errorf("Could not unmarshall serialized genome. %s", err.Error())
	}

	if nn.LRate != nn2.LRate {
		t.Errorf("LRate does not match")
	}

	if nn.TFunc != nn2.TFunc {
		t.Errorf("TFunc does not match")
	}

	if len(nn.NeuralLayers) != len(nn2.NeuralLayers) {
		t.Errorf("Len() of NeuralLayers does not match")
	}

	for i := range nn.NeuralLayers {
		if nn.NeuralLayers[i].Length != nn2.NeuralLayers[i].Length {
			t.Errorf("Length attr of layer %d does not match", i)
		}

		if len(nn.NeuralLayers[i].NeuronUnits) != len(nn2.NeuralLayers[i].NeuronUnits) {
			t.Errorf("Actual length of neurons of layer %d does not match", i)
		}

		for j := range nn.NeuralLayers[i].NeuronUnits {
			if nn.NeuralLayers[i].NeuronUnits[j].Lrate != nn2.NeuralLayers[i].NeuronUnits[j].Lrate {
				t.Errorf("Lrate of neuron [%d, %d] does not match", i, j)
			}

			if nn.NeuralLayers[i].NeuronUnits[j].Bias != nn2.NeuralLayers[i].NeuronUnits[j].Bias {
				t.Errorf("Bias of neuron [%d, %d] does not match", i, j)
			}

			if nn.NeuralLayers[i].NeuronUnits[j].Value != nn2.NeuralLayers[i].NeuronUnits[j].Value {
				t.Errorf("Value of neuron [%d, %d] does not match", i, j)
			}

			if nn.NeuralLayers[i].NeuronUnits[j].Delta != nn2.NeuralLayers[i].NeuronUnits[j].Delta {
				t.Errorf("Value of neuron [%d, %d] does not match", i, j)
			}

			if len(nn.NeuralLayers[i].NeuronUnits[j].Weights) != len(nn2.NeuralLayers[i].NeuronUnits[j].Weights) {
				t.Errorf("Length of weights of neuron [%d, %d] does not match", i, j)
			}

			for k := range nn.NeuralLayers[i].NeuronUnits[j].Weights {
				if nn.NeuralLayers[i].NeuronUnits[j].Weights[k] != nn2.NeuralLayers[i].NeuronUnits[j].Weights[k] {
					t.Errorf("Weigth does not match")
				}
			}
		}

	}
}

// DeserializeGenome deserialized buff to a MLP
func TestDeserializeGenome(t *testing.T) {
	nn := &mlp.MultiLayerNetwork{}
	size := 5

	nn.LRate = 0.5
	nn.NeuralLayers = make([]mlp.NeuralLayer, 3)
	nn.NeuralLayers[1].Length = int64(size)
	nn.NeuralLayers[1].NeuronUnits = make([]mlp.NeuronUnit, size)
	nn.NeuralLayers[2].Length = 2
	nn.NeuralLayers[2].NeuronUnits = make([]mlp.NeuronUnit, 2)
	nn.NeuralLayers[1].NeuronUnits[1].Bias = 0.1
	nn.NeuralLayers[1].NeuronUnits[1].Value = 0.2
	nn.NeuralLayers[1].NeuronUnits[1].Lrate = 0.3
	nn.NeuralLayers[1].NeuronUnits[1].Delta = 0.4
	nn.NeuralLayers[1].NeuronUnits[1].Weights = make([]float64, 2)
	nn.NeuralLayers[1].NeuronUnits[1].Weights[0] = 0.01
	nn.NeuralLayers[1].NeuronUnits[1].Weights[1] = 0.02

	buff, err := nn.Marshal()
	if err != nil {
		t.Errorf("Could not serialize MultiLayerNetwork. %s", err.Error())
	}

	delegate := mlp.DelegateImpl{}
	genome := delegate.DeserializeGenome(buff)
	nn2 := genome.(*mlp.MultiLayerNetwork)

	if nn.LRate != nn2.LRate {
		t.Errorf("LRate does not match")
	}

	if nn.TFunc != nn2.TFunc {
		t.Errorf("TFunc does not match")
	}

	if len(nn.NeuralLayers) != len(nn2.NeuralLayers) {
		t.Errorf("Len() of NeuralLayers does not match")
	}

	for i := range nn.NeuralLayers {
		if nn.NeuralLayers[i].Length != nn2.NeuralLayers[i].Length {
			t.Errorf("Length attr of layer %d does not match", i)
		}

		if len(nn.NeuralLayers[i].NeuronUnits) != len(nn2.NeuralLayers[i].NeuronUnits) {
			t.Errorf("Actual length of neurons of layer %d does not match", i)
		}

		for j := range nn.NeuralLayers[i].NeuronUnits {
			if nn.NeuralLayers[i].NeuronUnits[j].Lrate != nn2.NeuralLayers[i].NeuronUnits[j].Lrate {
				t.Errorf("Lrate of neuron [%d, %d] does not match", i, j)
			}

			if nn.NeuralLayers[i].NeuronUnits[j].Bias != nn2.NeuralLayers[i].NeuronUnits[j].Bias {
				t.Errorf("Bias of neuron [%d, %d] does not match", i, j)
			}

			if nn.NeuralLayers[i].NeuronUnits[j].Value != nn2.NeuralLayers[i].NeuronUnits[j].Value {
				t.Errorf("Value of neuron [%d, %d] does not match", i, j)
			}

			if nn.NeuralLayers[i].NeuronUnits[j].Delta != nn2.NeuralLayers[i].NeuronUnits[j].Delta {
				t.Errorf("Value of neuron [%d, %d] does not match", i, j)
			}

			if len(nn.NeuralLayers[i].NeuronUnits[j].Weights) != len(nn2.NeuralLayers[i].NeuronUnits[j].Weights) {
				t.Errorf("Length of weights of neuron [%d, %d] does not match", i, j)
			}

			for k := range nn.NeuralLayers[i].NeuronUnits[j].Weights {
				if nn.NeuralLayers[i].NeuronUnits[j].Weights[k] != nn2.NeuralLayers[i].NeuronUnits[j].Weights[k] {
					t.Errorf("Weigth does not match")
				}
			}
		}

	}
}
