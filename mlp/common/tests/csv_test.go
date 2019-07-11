package common

import (
	"testing"

	"github.com/salvacorts/go-perceptron-go/model/neural"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	mlpLogger "github.com/sirupsen/logrus"
)

func TestLoadPatternsFromCSVFile(t *testing.T) {
	csv :=
		`
5.1,3.4,1.5,0.2,Iris-setosa
6.3,2.5,5,1.9,Iris-virginica
5.9,3,4.2,1.5,Iris-versicolor`

	mlpLogger.SetLevel(mlpLogger.ErrorLevel)
	pattern, err, mapped := utils.LoadPatternsFromCSV(csv)
	if err != nil {
		t.Errorf("Error loading CSV. Error: %s", err.Error())
	}

	expectedMapped := []string{
		"Iris-setosa",
		"Iris-virginica",
		"Iris-versicolor"}

	if !equalString(mapped, expectedMapped) {
		t.Errorf("Mapped categories does not match")
	}

	expectedPattern := []neural.Pattern{
		neural.Pattern{
			Features:             []float64{5.1, 3.4, 1.5, 0.2},
			SingleRawExpectation: "Iris-setosa",
			SingleExpectation:    0,
			MultipleExpectation:  []float64{}},
		neural.Pattern{
			Features:             []float64{6.3, 2.5, 5, 1.9},
			SingleRawExpectation: "Iris-virginica",
			SingleExpectation:    1,
			MultipleExpectation:  []float64{}},
		neural.Pattern{
			Features:             []float64{5.9, 3, 4.2, 1.5},
			SingleRawExpectation: "Iris-versicolor",
			SingleExpectation:    2,
			MultipleExpectation:  []float64{}},
	}

	if !equalPattern(pattern, expectedPattern) {
		t.Errorf("Pattens does not match")
	}
}
