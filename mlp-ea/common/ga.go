package common

import (
	"time"

	utils "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	"github.com/salvacorts/eaopt"
	mn "github.com/salvacorts/go-perceptron-go/model/neural"
	mv "github.com/salvacorts/go-perceptron-go/validation"
	"github.com/sirupsen/logrus"
)

// Log is the logger instance for the GA
var Log = logrus.New()

// Config sets the Training configuration for the chormosomes
var Config MLPConfig

// TrainMLP trains a Multi Layer Perceptron
func TrainMLP(csvdata string) (mn.MultiLayerNetwork, float64, error) {
	start := time.Now()

	// Patterns initialization
	var patterns, _, mapped = utils.LoadPatternsFromCSV(csvdata)
	train, test := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	ga, err := eaopt.NewDefaultGAConfig().NewGA()
	if err != nil {
		return mn.MultiLayerNetwork{}, 0, err
	}

	// Configure ga
	ga.NGenerations = generations
	ga.NPops = 1
	ga.PopSize = popSize
	ga.Model = getGenerationalModelRoulette()
	ga.Callback = func(ga *eaopt.GA) {
		Log.WithFields(logrus.Fields{
			"level":               "info",
			"Generation":          ga.Generations,
			"Avg":                 ga.Populations[0].Individuals.FitAvg(),
			"Fitness":             ga.HallOfFame[0].Fitness,
			"HiddenLayer_Neurons": ga.HallOfFame[0].Genome.(*MLP).NeuralLayers[1].Length,
		}).Infof("Best fitness at generation %d: %f", ga.Generations, ga.HallOfFame[0].Fitness)
	}

	ga.EarlyStop = func(ga *eaopt.GA) bool {
		return ga.HallOfFame[0].Fitness == 0.0
	}

	// Configure MLP
	Config = MLPConfig{
		Epochs:      10,
		Folds:       1,
		Classes:     &mapped,
		TrainingSet: &train,
		FactoryCfg: MLPFactoryConfig{
			InputLayers:      len(patterns[0].Features),
			OutputLayers:     len(mapped),
			MinHiddenNeurons: 2,
			MaxHiddenNeurons: 20,
			Tfunc:            mn.SigmoidalTransfer,
			TfuncDeriv:       mn.SigmoidalTransferDerivate,
			MaxLR:            0.3,
			MinLR:            0.01,
		},
	}

	// Execute GA
	err = ga.Minimize(NewRandMLP)
	if err != nil {
		return mn.MultiLayerNetwork{}, 0, err
	}

	Log.WithFields(logrus.Fields{
		"level":    "info",
		"ExecTime": time.Since(start),
	}).Infof("Execution time: %s", time.Since(start))

	best := ga.HallOfFame[0].Genome.(*MLP)
	bestScore := ga.HallOfFame[0].Fitness

	Log.WithFields(logrus.Fields{
		"level":           "info",
		"TrainingFitness": bestScore,
	}).Infof("Training Error: %f", bestScore)

	predictions := utils.PredictN((*mn.MultiLayerNetwork)(best), test)
	predictionsR := utils.RoundPredictions(predictions)
	_, testAcc := utils.AccuracyN(predictionsR, test)

	Log.WithFields(logrus.Fields{
		"level":       "info",
		"TestFitness": 100 - testAcc,
	}).Infof("Test Error: %f", 100-testAcc)

	return mn.MultiLayerNetwork(*best), bestScore, nil
}
