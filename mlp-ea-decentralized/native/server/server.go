package main

import (
	"flag"
	"io/ioutil"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/ga"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	"github.com/salvacorts/eaopt"
	mv "github.com/salvacorts/go-perceptron-go/validation"
	"google.golang.org/grpc/grpclog"

	"github.com/sirupsen/logrus"
)

// TODO: Append here GA and Pool params
var (
	// Opened ports
	grpcPort        = flag.Int("grpcPort", 3117, "Port to listen for grpc requests")
	clusterPort     = flag.Int("clusterPort", 9999, "Listening port for gossip protocol")
	clusterBoostrap = flag.String("clusterBoostrap", "", "comma separated list of peers already ina  cluster to join")
)

func main() {
	flag.Parse()

	filename := "../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		logrus.Fatalf("Cannot open %s. Error: %s", filename, err.Error())
	}

	// Patterns initialization
	var patterns, _, mapped = utils.LoadPatternsFromCSV(string(fileContent))
	train, test := mv.TrainTestPatternSplit(patterns, 0.8, 1)

	// Configure MLP
	mlp.Config = mlp.MLPConfig{
		Epochs:      50,
		Folds:       5,
		Classes:     mapped,
		TrainingSet: train,
		TrainEpochs: 100,
		MutateRate:  0.3,
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

	// Create Pool
	pool := ga.MakePool(
		100, *grpcPort, *clusterPort,
		strings.Split(*clusterBoostrap, ","),
		//rand.New(rand.NewSource(7)))
		rand.New(rand.NewSource(time.Now().Unix())))

	// Configure  extra pool settings
	pool.Delegate = mlp.DelegateImpl{}
	pool.KeepBest = false
	pool.NMigrate = pool.PopSize / 10
	pool.SortPrecission = 100
	pool.SortFunc = mlp.SortByFitnessAndNeurons
	pool.CrossRate = 0.3
	pool.MutRate = 0.3
	pool.MaxEvaluations = 1000000
	pool.ExtraOperators = []eaopt.ExtraOperator{
		eaopt.ExtraOperator{Operator: mlp.AddNeuron, Probability: 0.3},
		eaopt.ExtraOperator{Operator: mlp.RemoveNeuron, Probability: 0.15},
		eaopt.ExtraOperator{Operator: mlp.SubstituteNeuron, Probability: 0.15},
		eaopt.ExtraOperator{Operator: mlp.Train, Probability: 0.3},
	}
	pool.BestCallback = func(pool *ga.PoolModel) {
		logrus.WithFields(logrus.Fields{
			"level":       "info",
			"Evaluations": pool.GetTotalEvaluations(),
			"Fitness":     pool.BestSolution.Fitness,
			"ID":          pool.BestSolution.ID,
		}).Infof("New best solution with fitness: %f", pool.BestSolution.Fitness)
	}
	pool.EarlyStop = func(pool *ga.PoolModel) bool {
		return pool.BestSolution.Fitness == 0
	}

	// Configure Logs
	logrus.SetOutput(os.Stdout)
	logrus.SetLevel(logrus.InfoLevel)
	ga.Log.SetOutput(os.Stdout)
	ga.Log.SetLevel(logrus.InfoLevel)
	grpclog.SetLoggerV2(grpclog.NewLoggerV2(ga.Log.Out, ga.Log.Out, ga.Log.Out))

	start := time.Now()

	// Execute pool-based GA
	pool.Minimize()

	logrus.WithFields(logrus.Fields{
		"level":    "info",
		"ExecTime": time.Since(start),
	}).Infof("Execution time: %s", time.Since(start))

	best := pool.BestSolution.Genome.(*mlp.MultiLayerNetwork)
	bestScore := pool.BestSolution.Fitness

	logrus.WithFields(logrus.Fields{
		"level":           "info",
		"TrainingFitness": bestScore,
	}).Infof("Training Error: %f", bestScore)

	predictions := mlp.PredictN(best, test)
	predictionsR := utils.RoundPredictions(predictions)
	_, testAcc := utils.AccuracyN(predictionsR, test)

	logrus.WithFields(logrus.Fields{
		"level":       "info",
		"TestFitness": 100 - testAcc,
	}).Infof("Test Error: %f", 100-testAcc)
}
