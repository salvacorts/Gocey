package common

import (
	"testing"

	"github.com/made2591/go-perceptron-go/model/neural"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common"
	mlpLogger "github.com/sirupsen/logrus"
)

func equalString(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func equalFloat64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func equalPattern(a, b []neural.Pattern) bool {
	if len(a) != len(b) {
		return false
	}

	for i := 0; i < len(a); i++ {
		if a[i].SingleExpectation != b[i].SingleExpectation {
			return false
		}

		if a[i].SingleRawExpectation != b[i].SingleRawExpectation {
			return false
		}

		if !equalFloat64(a[i].MultipleExpectation, b[i].MultipleExpectation) {
			return false
		}

		if !equalFloat64(a[i].Features, b[i].Features) {
			return false
		}
	}

	return true
}

func TestLoadPatternsFromCSVFile(t *testing.T) {
	csv :=
		`
5.1,3.4,1.5,0.2,Iris-setosa
6.3,2.5,5,1.9,Iris-virginica
5.9,3,4.2,1.5,Iris-versicolor`

	mlpLogger.SetLevel(mlpLogger.ErrorLevel)
	pattern, err, mapped := common.LoadPatternsFromCSV(csv)
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
