package common

import (
	"log"
	"time"

	eaopt "github.com/MaxHalford/eaopt"
	mn "github.com/made2591/go-perceptron-go/model/neural"
	utils "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common"
)

// TrainMLP trains a Multi Layer Perceptron
func TrainMLP(csvdata string) (mn.MultiLayerNetwork, float64, error) {
	var start = time.Now()

	// Patterns initialization
	var patterns, _, mapped = utils.LoadPatternsFromCSV(csvdata)

	ga, err := eaopt.NewDefaultGAConfig().NewGA()
	if err != nil {
		return mn.MultiLayerNetwork{}, 0, err
	}

	// Configure ga
	ga.NGenerations = 10
	ga.NPops = 1
	ga.PopSize = 50
	ga.Callback = func(ga *eaopt.GA) {
		log.Printf("Best fitness at generation %d: %f\n", ga.Generations, ga.HallOfFame[0].Fitness)
	}

	// Congigure MLP Factory
	mlpFactory := MLPFactory{
		InputLayers:     len(patterns[0].Features),
		OutputLayers:    len(mapped),
		MaxHiddenLayers: 30,
		Tfunc:           mn.SigmoidalTransfer,
		TfuncDeriv:      mn.SigmoidalTransferDerivate,
		MaxLR:           0.3,
		MinLR:           0.1,

		Config: MLPConfig{
			Epochs:    1,
			Classes:   &mapped,
			TrainData: &patterns,
			TestData:  &patterns, // TODO: Split data pattern
		},
	}

	// Execute GA
	err = ga.Minimize(mlpFactory.NewRandMLP)
	if err != nil {
		return mn.MultiLayerNetwork{}, 0, err
	}

	log.Printf("Execution time: %s\n", time.Since(start))

	best := ga.HallOfFame[0].Genome.(MLP)
	bestScore := ga.HallOfFame[0].Fitness

	return best.nn, bestScore, nil
}
