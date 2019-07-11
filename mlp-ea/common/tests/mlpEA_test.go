package common

import (
	"io/ioutil"
	"os"
	"testing"

	ga "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea/common"
	"github.com/sirupsen/logrus"
)

func TestTrainMLP(t *testing.T) {
	filename := "../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Errorf("Cannot open %s. Error: %s", filename, err.Error())
	}

	logrus.SetOutput(os.Stdout)
	logrus.SetLevel(logrus.ErrorLevel)
	ga.Log.SetOutput(os.Stdout)
	ga.Log.SetLevel(logrus.InfoLevel)
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
