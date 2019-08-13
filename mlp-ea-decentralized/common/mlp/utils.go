package mlp

import (
	mn "github.com/salvacorts/go-perceptron-go/model/neural"
	"github.com/salvacorts/go-perceptron-go/util"
	"github.com/salvacorts/go-perceptron-go/validation"
)

// Remove the given index i from the slice of neurons
func Remove(slice []NeuronUnit, i int) []NeuronUnit {
	return append(slice[:i], slice[i+1:]...)
}

func KFoldValidation(mlp *MultiLayerNetwork, patterns []mn.Pattern, epochs int, k int, shuffle int, mapped []string) ([]float64, float64) {

	// results and predictions vars init
	var scores, actual, predicted []float64
	var train, test []mn.Pattern

	scores = make([]float64, k)

	// split the dataset with shuffling
	folds := validation.KFoldPatternsSplit(patterns, k, shuffle)

	// the t-th fold is used as test
	for t := 0; t < k; t++ {
		// prepare train
		train = nil
		for i := 0; i < k; i++ {
			if i != t {
				train = append(train, folds[i]...)
			}
		}
		test = folds[t]

		// train mlp with set of patterns, for specified number of epochs
		Training(mlp, patterns, mapped, epochs)

		var tFunc TransferF

		switch mlp.TFunc {
		case TransferFunc_SIGMOIDAL:
			tFunc = mn.SigmoidalTransfer
		}

		// compute predictions for each pattern in testing set
		for _, pattern := range test {
			// get actual
			actual = append(actual, pattern.SingleExpectation)
			// get output from network
			o_out := Execute(mlp, &pattern, tFunc)
			// get index of max output
			_, indexMaxOut := util.MaxInSlice(o_out)
			// add to predicted values
			predicted = append(predicted, float64(indexMaxOut))
		}

		// compute score
		_, percentageCorrect := mn.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		// log.WithFields(log.Fields{
		// 	"level":             "info",
		// 	"place":             "validation",
		// 	"method":            "MLPKFoldValidation",
		// 	"foldNumber":        t,
		// 	"trainSetLen":       len(train),
		// 	"testSetLen":        len(test),
		// 	"percentageCorrect": percentageCorrect,
		// }).Info("Evaluation completed for current fold.")
	}

	// compute average score
	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}

	mean := acc / float64(len(scores))

	// log.WithFields(log.Fields{
	// 	"level":       "info",
	// 	"place":       "validation",
	// 	"method":      "MLPKFoldValidation",
	// 	"folds":       k,
	// 	"trainSetLen": len(train),
	// 	"testSetLen":  len(test),
	// 	"meanScore":   mean,
	// }).Info("Evaluation completed for all folds.")

	return scores, mean
}

// PredictN the output for a set of patterns
func PredictN(mlp *MultiLayerNetwork, input []mn.Pattern) (out [][]float64) {
	out = make([][]float64, len(input))

	var (
		tFunc TransferF
	)

	switch mlp.TFunc {
	case TransferFunc_SIGMOIDAL:
		tFunc = mn.SigmoidalTransfer
		// TODO: Add others here
	}

	for i, pattern := range input {
		out[i] = Execute(mlp, &pattern, tFunc)
	}

	return out
}
