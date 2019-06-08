package gomlp_test

import (
	"fmt"
	"math/rand"
	"testing"

	gomlp "github.com/salvacorts/TFG-Parasitic-Metaheuristics/gomlp"
	transf "github.com/salvacorts/TFG-Parasitic-Metaheuristics/gomlp/transferfunctions"
	utils "github.com/salvacorts/TFG-Parasitic-Metaheuristics/gomlp/utils"
)

func TestGlass(t *testing.T) {
	layers := []int{10, 5, 4, 7}
	sigm := transf.MakeSigmoidalTransferFunc()

	mlp := gomlp.MakeMLP(layers, 0.6, sigm)

	// Read data and shuffle it
	data := utils.LoadCSV("datasets/glass.data")
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})

	X := make([][]float64, len(data))
	y := make([]float64, len(data))

	for i := 0; i < len(data); i++ {
		X[i] = make([]float64, len(data[i])-1)
		X[i] = data[i][:len(data[i])-1]

		y[i] = data[i][len(data[i])-1]
	}

	y_onehot := utils.OneHotEncode(y, 7)

	// Divide train and test
	n_train := int(float64(len(X)) * 0.8)
	X_train := X[0:n_train]
	y_train := y_onehot[0:n_train]
	X_test := X[n_train:]
	y_test := y_onehot[n_train:]

	// Train
	mlp.Train(X_train, y_train)

	// Predict
	predictions := mlp.PredictN(X_test)

	// Get the one with maximum
	for i := 0; i < len(predictions); i++ {
		max_idx := 0
		max := 0.0

		for j := 0; j < len(predictions[i]); j++ {
			if predictions[i][j] > max {
				max = predictions[i][j]
				max_idx = j
			}
		}

		for j := 0; j < len(predictions[i]); j++ {
			if j != max_idx {
				predictions[i][j] = 0
			} else {
				predictions[i][j] = 1
			}
		}
	}

	accuracy := utils.GetAccuracy(predictions, y_test)
	fmt.Printf("Accuracy: %f", accuracy)
}
