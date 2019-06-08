package main

import (
	mn "github.com/made2591/go-perceptron-go/model/neural"
	v "github.com/made2591/go-perceptron-go/validation"
	log "github.com/sirupsen/logrus"
)

func main() {
	log.WithFields(log.Fields{
		"level": "info",
		"place": "main",
		"msg":   "multi layer perceptron train and test over iris dataset",
	}).Info("Compute backpropagation multi layer perceptron on sonar data set (binary classification problem)")

	// percentage and shuffling in dataset
	var filePath = "./iris.csv"
	//filePath = "./res/sonar.all_data.csv"

	// single layer neuron parameters
	var learningRate = 0.01
	var percentage = 0.67
	var shuffle = 1

	// training parameters
	var epochs = 500
	var folds = 3

	// Patterns initialization
	var patterns, _, mapped = mn.LoadPatternsFromCSVFile(filePath)

	//input  layer : 4 neuron, represents the feature of Iris, more in general dimensions of pattern
	//hidden layer : 3 neuron, activation using sigmoid, number of neuron in hidden level
	// 2° hidden l : * neuron, insert number of level you want
	//output layer : 3 neuron, represents the class of Iris, more in general dimensions of mapped values
	var layers = []int{len(patterns[0].Features), 20, len(mapped)}

	//Multilayer perceptron model, with one hidden layer.
	var mlp mn.MultiLayerNetwork = mn.PrepareMLPNet(layers, learningRate, mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)

	// compute scores for each folds execution
	var scores = v.MLPKFoldValidation(&mlp, patterns, epochs, folds, shuffle, mapped)

	// use simpler validation
	var mlp2 mn.MultiLayerNetwork = mn.PrepareMLPNet(layers, learningRate, mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)
	var scores2 = v.MLPRandomSubsamplingValidation(&mlp2, patterns, percentage, epochs, folds, shuffle, mapped)

	log.WithFields(log.Fields{
		"level":  "info",
		"place":  "main",
		"scores": scores,
	}).Info("Scores reached: ", scores)

	log.WithFields(log.Fields{
		"level":  "info",
		"place":  "main",
		"scores": scores2,
	}).Info("Scores reached: ", scores2)
}
