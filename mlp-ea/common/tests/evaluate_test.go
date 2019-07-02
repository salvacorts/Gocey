package common

import (
	"io/ioutil"
	"math/rand"
	"testing"

	mn "github.com/made2591/go-perceptron-go/model/neural"
	v "github.com/made2591/go-perceptron-go/validation"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea/common"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
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
			Classes:       &mapped,
			TrainingSet:   &train,
			ValidationSet: &validation,
		},
	}

	rnd := rand.New(rand.NewSource(7))
	genome := mlpFactory.NewRandMLP(rnd)

	score, err := genome.Evaluate()
	if err != nil {
		t.Errorf("Error training MLP: %s", err.Error())
	}

	expectedScore := 0.0
	if score < expectedScore {
		t.Errorf("Got training score (%f) under threshold (%f)", score, expectedScore)
	}
}
