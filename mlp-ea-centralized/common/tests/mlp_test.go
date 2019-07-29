package tests

import (
	"io/ioutil"
	"math/rand"
	"os"
	"testing"

	"github.com/gogo/protobuf/proto"

	ga "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common/mlp"
	"github.com/sirupsen/logrus"
)

func TestMarshallUnmarshall(t *testing.T) {
	rnd := rand.New(rand.NewSource(7))

	ga.Config = ga.MLPConfig{
		FactoryCfg: ga.MLPFactoryConfig{
			InputLayers:      9,
			OutputLayers:     2,
			MinHiddenNeurons: 2,
			MaxHiddenNeurons: 20,
			Tfunc:            mlp.TransferFunc_SIGMOIDAL,
			MaxLR:            0.3,
			MinLR:            0.01,
		},
	}

	genome := ga.NewRandMLP(rnd)

	mlp1 := genome.(*ga.MLP)

	buff, err := proto.Marshal((*mlp.MultiLayerNetwork)(mlp1))
	if err != nil {
		t.Fatalf("Cannot marshall")
	}

	mlp2 := &mlp.MultiLayerNetwork{}
	err = proto.Unmarshal(buff, mlp2)
	if err != nil {
		t.Fatalf("Could not unmarshall")
	}

	mlp2.LRate++
	if mlp2.LRate == mlp1.LRate {
		t.Errorf("Modifiying one modifies the other (1st level)")
	}

	mlp2.NeuralLayers[1].Length++
	if mlp2.NeuralLayers[1].Length == mlp1.NeuralLayers[1].Length {
		t.Errorf("Modifiying one modifies the other (2nd level)")
	}

	mlp2.NeuralLayers[1].NeuronUnits[0].Bias++
	if mlp2.NeuralLayers[1].NeuronUnits[0].Bias == mlp1.NeuralLayers[1].NeuronUnits[0].Bias {
		t.Errorf("Modifiying one modifies the other (3rd level)")
	}

	mlp2.NeuralLayers[1].NeuronUnits[0].Weights[0]++
	if mlp2.NeuralLayers[1].NeuronUnits[0].Weights[0] == mlp1.NeuralLayers[1].NeuronUnits[0].Weights[0] {
		t.Errorf("Modifiying one modifies the other (4th level)")
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
	ga.Log.SetLevel(logrus.DebugLevel)

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
