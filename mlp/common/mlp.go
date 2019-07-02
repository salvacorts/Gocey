package common

import (
	"log"
	"time"

	mn "github.com/made2591/go-perceptron-go/model/neural"
	v "github.com/made2591/go-perceptron-go/validation"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
)

// TrainMLP trains a Multi Layer Perceptron
func TrainMLP(csvdata string) (mn.MultiLayerNetwork, []float64) {
	var start = time.Now()

	// single layer neuron parameters
	var learningRate = 0.01
	var shuffle = 1

	// training parameters
	var epochs = 1
	var folds = 2

	// Patterns initialization
	var patterns, _, mapped = utils.LoadPatternsFromCSV(csvdata)

	//input  layer : 4 neuron, represents the feature of Iris, more in general dimensions of pattern
	//hidden layer : 3 neuron, activation using sigmoid, number of neuron in hidden level
	// 2° hidden l : * neuron, insert number of level you want
	//output layer : 3 neuron, represents the class of Iris, more in general dimensions of mapped values
	var layers = []int{len(patterns[0].Features), 5, len(mapped)}

	//Multilayer perceptron model, with one hidden layer.
	var mlp = mn.PrepareMLPNet(layers, learningRate, mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)

	// compute scores for each folds execution
	var scores = v.MLPKFoldValidation(&mlp, patterns, epochs, folds, shuffle, mapped)

	log.Printf("Execution time: %s\n", time.Since(start))

	return mlp, scores
}
