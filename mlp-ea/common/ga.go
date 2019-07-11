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

// TrainMLP trains a Multi Layer Perceptron
func TrainMLP(csvdata string) (mn.MultiLayerNetwork, float64, error) {
	var start = time.Now()

	// Patterns initialization
	var patterns, _, mapped = utils.LoadPatternsFromCSV(csvdata)
	train, test := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	ga, err := eaopt.NewDefaultGAConfig().NewGA()
	if err != nil {
		return mn.MultiLayerNetwork{}, 0, err
	}

	// Configure ga
	ga.NGenerations = 100
	ga.NPops = 1
	ga.PopSize = 100
	ga.Model = eaopt.ModSteadyState{
		Selector:  eaopt.SelElitism{},
		KeepBest:  true,
		MutRate:   0.3,
		CrossRate: 0.3,

		ExtraOperators: []eaopt.ExtraOperator{
			eaopt.ExtraOperator{Operator: AddNeuron, Probability: 0.25},
			eaopt.ExtraOperator{Operator: RemoveNeuron, Probability: 0.2},
			eaopt.ExtraOperator{Operator: SubstituteNeuron, Probability: 0.25},
			eaopt.ExtraOperator{Operator: Train, Probability: 0.25},
		},
	}
	ga.Callback = func(ga *eaopt.GA) {
		Log.WithFields(logrus.Fields{
			"level":               "info",
			"Generation":          ga.Generations,
			"Fitness":             ga.HallOfFame[0].Fitness,
			"HiddenLayer_Neurons": ga.HallOfFame[0].Genome.(*MLP).NeuralLayers[1].Length,
		}).Infof("Best fitness at generation %d: %f", ga.Generations, ga.HallOfFame[0].Fitness)
	}

	// Configure MLP Factory
	mlpFactory := MLPFactory{
		InputLayers:      len(patterns[0].Features),
		OutputLayers:     len(mapped),
		MinHiddenNeurons: 2,
		MaxHiddenNeurons: 20,
		Tfunc:            mn.SigmoidalTransfer,
		TfuncDeriv:       mn.SigmoidalTransferDerivate,
		MaxLR:            0.3,
		MinLR:            0.01,
	}

	Config = MLPConfig{
		Epochs:      10,
		Folds:       1,
		Classes:     &mapped,
		TrainingSet: &train,
	}

	// Execute GA
	err = ga.Minimize(mlpFactory.NewRandMLP)
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
