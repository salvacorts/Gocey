package ga

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

// TrainMLP trains a Multi Layer Perceptron
func TrainMLP(csvdata string) (mlp.MultiLayerNetwork, float64, error) {
	start := time.Now()

	// Patterns initialization
	var patterns, _, mapped = utils.LoadPatternsFromCSV(csvdata)
	train, test := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	// Configure MLP
	mlp.Config = mlp.MLPConfig{
		Epochs:      50,
		Folds:       5,
		Classes:     mapped,
		TrainingSet: train,
		MutateRate:  mutProb,
		FactoryCfg: mlp.MLPFactoryConfig{
			InputLayers:      len(patterns[0].Features),
			OutputLayers:     len(mapped),
			MinHiddenNeurons: 2,
			MaxHiddenNeurons: 20,
			Tfunc:            mlp.TransferFunc_SIGMOIDAL,
			MaxLR:            0.3,
			MinLR:            0.01,
		},
	}

	pool := MakePool(popSize, rand.New(rand.NewSource(7)))

	// pool.Rnd = rand.New(rand.NewSource(7))
	pool.KeepBest = false
	pool.SortFunction = SortByFitnessAndNeurons
	// pool.PopSize = popSize
	pool.CrossRate = crossProb
	pool.MutRate = mutProb
	pool.MaxEvaluations = 1000000

	// pool.GenerationCallback = func(pool *PoolModel) {
	// 	Log.WithFields(logrus.Fields{
	// 		"level":               "info",
	// 		"Generation":          pool.Generation,
	// 		"Avg":                 pool.FitAvg(),
	// 		"Fitness":             pool.BestSolution.Fitness,
	// 		"HiddenLayer_Neurons": pool.BestSolution.Genome.(*MLP).NeuralLayers[1].Length,
	// 	}).Infof("Best fitness at generation %d: %f", pool.Generation, pool.BestSolution.Fitness)
	// }

	pool.BestCallback = func(pool *PoolModel) {
		Log.WithFields(logrus.Fields{
			"level":       "info",
			"Evaluations": pool.evaluations,
			"Fitness":     pool.BestSolution.Fitness,
			"ID":          pool.BestSolution.ID,
		}).Infof("New best solution with fitness: %f", pool.BestSolution.Fitness)
	}

	pool.ExtraOperators = []eaopt.ExtraOperator{
		eaopt.ExtraOperator{Operator: mlp.AddNeuron, Probability: addNeuronProb},
		eaopt.ExtraOperator{Operator: mlp.RemoveNeuron, Probability: removeNeuronProb},
		eaopt.ExtraOperator{Operator: mlp.SubstituteNeuron, Probability: substituteNeuronProb},
		eaopt.ExtraOperator{Operator: mlp.Train, Probability: trainProb},
	}

	pool.EarlyStop = func(pool *PoolModel) bool {
		return pool.BestSolution.Fitness == 0
	}

	// Execute pool-based GA
	pool.Minimize()

	Log.WithFields(logrus.Fields{
		"level":    "info",
		"ExecTime": time.Since(start),
	}).Infof("Execution time: %s", time.Since(start))

	best := pool.BestSolution.Genome.(*mlp.MLP)
	bestScore := pool.BestSolution.Fitness

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
