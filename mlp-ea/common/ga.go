package common

import (
	"time"

	mn "github.com/made2591/go-perceptron-go/model/neural"
	mv "github.com/made2591/go-perceptron-go/validation"
	utils "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	"github.com/salvacorts/eaopt"
	"github.com/sirupsen/logrus"
)

// Log is the logger instance for the GA
var Log = logrus.New()

// TrainMLP trains a Multi Layer Perceptron
func TrainMLP(csvdata string) (mn.MultiLayerNetwork, float64, error) {
	var start = time.Now()

	// Patterns initialization
	var patterns, _, mapped = utils.LoadPatternsFromCSV(csvdata)
	train, validation := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	ga, err := eaopt.NewDefaultGAConfig().NewGA()
	if err != nil {
		return mn.MultiLayerNetwork{}, 0, err
	}

	// Configure ga
	ga.NGenerations = 20
	ga.NPops = 1
	ga.PopSize = 20
	ga.Model = eaopt.ModSteadyState{
		Selector:  eaopt.SelElitism{},
		KeepBest:  true,
		MutRate:   0.4,
		CrossRate: 0.5,

		ExtraOperators: []eaopt.ExtraOperator{
			eaopt.ExtraOperator{Operator: AddNeuron, Probability: 0.1},
			eaopt.ExtraOperator{Operator: RemoveNeuron, Probability: 0.1},
			eaopt.ExtraOperator{Operator: SubstituteNeuron, Probability: 0.1},
			eaopt.ExtraOperator{Operator: Train, Probability: 0.1},
		},
	}
	ga.Callback = func(ga *eaopt.GA) {
		Log.WithFields(logrus.Fields{
			"level":      "info",
			"Generation": ga.Generations,
			"Fitness":    ga.HallOfFame[0].Fitness,
		}).Infof("Best fitness at generation %d: %f", ga.Generations, ga.HallOfFame[0].Fitness)
	}

	// Configure MLP Factory
	mlpFactory := MLPFactory{
		InputLayers:     len(patterns[0].Features),
		OutputLayers:    len(mapped),
		MinHiddenLayers: 2,
		MaxHiddenLayers: 20,
		Tfunc:           mn.SigmoidalTransfer,
		TfuncDeriv:      mn.SigmoidalTransferDerivate,
		MaxLR:           0.3,
		MinLR:           0.01,

		Config: MLPConfig{
			Epochs:        10,
			Folds:         1,
			Classes:       &mapped,
			TrainingSet:   &train,
			ValidationSet: &validation,
		},
	}

	// Execute GA
	err = ga.Minimize(mlpFactory.NewRandMLP)
	if err != nil {
		return mn.MultiLayerNetwork{}, 0, err
	}

	Log.WithFields(logrus.Fields{
		"level":    "info",
		"ExecTime": time.Since(start),
	}).Infof("Execution time: %s\n", time.Since(start))

	best := ga.HallOfFame[0].Genome.(MLP)
	bestScore := ga.HallOfFame[0].Fitness

	return best.NeuralNet, bestScore, nil
}
