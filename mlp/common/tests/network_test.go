package common

import (
	"io/ioutil"
	"testing"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	mlpLogger "github.com/sirupsen/logrus"
)

func TestRoundPredictions(t *testing.T) {

	input := [][]float64{
		[]float64{0.12, 0.11, 0.76, 0.4},
		[]float64{0.0, 0.333, 0.99, 0.999},
		[]float64{0.67, 0.47, 0.56, 0.12},
	}

	expected := []float64{2, 3, 0}

	output := utils.RoundPrediction(input[0])

	if float64(output) != expected[0] {
		t.Errorf("Rounded and expected does not match:\n\tRounded: %d\n\tExpected: %f", output, expected[0])
	}

	outputN := utils.RoundPredictions(input)

	if !equalFloat64(outputN, expected) {
		t.Errorf("Multiple Rounded and expected does not match:\n\tRounded: %v\n\tExpected: %v", outputN, expected)
	}
}

func TestAccuracy(t *testing.T) {
	filename := "../../../datasets/iris.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Errorf("Cannot open %s. Error: %s", filename, err.Error())
	}

	mlpLogger.SetLevel(mlpLogger.ErrorLevel)

	var patterns, _, _ = utils.LoadPatternsFromCSV(string(fileContent))

	actual := patterns[:4]

	allGood := []float64{0, 1, 2, 2}
	_, acc := utils.AccuracyN(allGood, actual)
	if acc != 100 {
		t.Errorf("Expected accuracy 100, got %f", acc)
	}

	allBad := []float64{7, 7, 7, 7}
	_, acc = utils.AccuracyN(allBad, actual)
	if acc != 0 {
		t.Errorf("Expected accuracy 0, got %f", acc)
	}

	halfGood := []float64{7, 1, 2, 7}
	_, acc = utils.AccuracyN(halfGood, actual)
	if acc != 50 {
		t.Errorf("Expected accuracy 50, got %f", acc)
	}

	somegood := []float64{7, 1, 7, 7}
	_, acc = utils.AccuracyN(somegood, actual)
	if acc != 25 {
		t.Errorf("Expected accuracy 25, got %f", acc)
	}

	mostgood := []float64{0, 1, 7, 2}
	_, acc = utils.AccuracyN(mostgood, actual)
	if acc != 75 {
		t.Errorf("Expected accuracy 75, got %f", acc)
	}
}
