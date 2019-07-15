package common

import (
	"io/ioutil"
	"testing"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common"
	mlpLogger "github.com/sirupsen/logrus"
)

func TestTrainMLP(t *testing.T) {
	filename := "../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Errorf("Cannot open %s. Error: %s", filename, err.Error())
	}

	mlpLogger.SetLevel(mlpLogger.ErrorLevel)
	_, score := common.TrainMLP(string(fileContent))

	expectedScore := 28.0

	if score < expectedScore {
		t.Errorf("Got training score (%f) under threshold (%f)", score, expectedScore)
	}
}
