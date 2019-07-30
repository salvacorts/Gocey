package common

import (
	"math/rand"
	"time"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common/mlp"
	utils "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	"github.com/salvacorts/eaopt"
	mv "github.com/salvacorts/go-perceptron-go/validation"
	"github.com/sirupsen/logrus"
)

// Log is the logger instance for the GA
var Log = logrus.New()

// Config sets the Training configuration for the chormosomes
var Config MLPConfig

// TrainMLP trains a Multi Layer Perceptron
func TrainMLP(csvdata string) (mlp.MultiLayerNetwork, float64, error) {
	start := time.Now()

	// Patterns initialization
	var patterns, _, mapped = utils.LoadPatternsFromCSV(csvdata)
	train, test := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	ga := MakePool()

	ga.Rnd = rand.New(rand.NewSource(7))
	ga.KeepBest = false
	ga.SortFunction = SortByFitnessAndNeurons
	ga.PopSize = popSize
	ga.CrossRate = crossProb
	ga.MutRate = mutProb
	ga.MaxGeneration = generations

	ga.GenerationCallback = func(ga *PipedPoolModel) {
		Log.WithFields(logrus.Fields{
			"level":               "info",
			"Generation":          ga.Generation,
			"Avg":                 ga.FitAvg(),
			"Fitness":             ga.BestSolution.Fitness,
			"HiddenLayer_Neurons": ga.BestSolution.Genome.(*MLP).NeuralLayers[1].Length,
		}).Infof("Best fitness at generation %d: %f", ga.Generation, ga.BestSolution.Fitness)
	}

	ga.BestCallback = func(ga *PipedPoolModel) {
		Log.WithFields(logrus.Fields{
			"level":      "info",
			"Generation": ga.Generation,
			"Fitness":    ga.BestSolution.Fitness,
		}).Infof("New best solution with fitness: %f", ga.BestSolution.Fitness)
	}

	ga.ExtraOperators = []eaopt.ExtraOperator{
		eaopt.ExtraOperator{Operator: AddNeuron, Probability: addNeuronProb},
		eaopt.ExtraOperator{Operator: RemoveNeuron, Probability: removeNeuronProb},
		eaopt.ExtraOperator{Operator: SubstituteNeuron, Probability: substituteNeuronProb},
		eaopt.ExtraOperator{Operator: Train, Probability: trainProb},
	}

	ga.EarlyStop = func(ga *PipedPoolModel) bool {
		return ga.BestSolution.Fitness == 0
	}

	// Configure MLP
	Config = MLPConfig{
		Epochs:      10,
		Folds:       1,
		Classes:     mapped,
		TrainingSet: train,
		FactoryCfg: MLPFactoryConfig{
			InputLayers:      len(patterns[0].Features),
			OutputLayers:     len(mapped),
			MinHiddenNeurons: 2,
			MaxHiddenNeurons: 20,
			Tfunc:            mlp.TransferFunc_SIGMOIDAL,
			MaxLR:            0.3,
			MinLR:            0.01,
		},
	}

	// Execute GA
	ga.Minimize()

	Log.WithFields(logrus.Fields{
		"level":    "info",
		"ExecTime": time.Since(start),
	}).Infof("Execution time: %s", time.Since(start))

	best := ga.BestSolution.Genome.(*MLP)
	bestScore := ga.BestSolution.Fitness

	Log.WithFields(logrus.Fields{
		"level":           "info",
		"TrainingFitness": bestScore,
	}).Infof("Training Error: %f", bestScore)

	predictions := mlp.PredictN((*mlp.MultiLayerNetwork)(best), test)
	predictionsR := utils.RoundPredictions(predictions)
	_, testAcc := utils.AccuracyN(predictionsR, test)

	Log.WithFields(logrus.Fields{
		"level":       "info",
		"TestFitness": 100 - testAcc,
	}).Infof("Test Error: %f", 100-testAcc)

	return mlp.MultiLayerNetwork(*best), bestScore, nil
}
