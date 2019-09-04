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

	"github.com/sirupsen/logrus"
)

// TODO: Append here GA and Pool params
var (
	// Opened ports
	metricsPort     = flag.Int("metricsPort", 2222, "Port for prometheus metrics")
	grpcPort        = flag.Int("grpcPort", 2019, "Port to listen for grpc requests")
	clusterPort     = flag.Int("clusterPort", 9999, "Listening port for gossip protocol")
	clusterBoostrap = flag.String("clusterBootstrap", "", "comma separated list of peers already in a cluster to join")
	dataset         = flag.String("datasetPath", "../../../datasets/glass.csv", "Dataset to train MLP with")
	webPath         = flag.String("webPath", "../../web/src/", "Webpage source code that will be served to browser-based collaborators")
	webPort         = flag.Int("webPort", 8080, "Port to serve webpage to broser based collaborators")
)

func main() {
	flag.Parse()

	fileContent, err := ioutil.ReadFile(*dataset)
	if err != nil {
		logrus.Fatalf("Cannot open %s. Error: %s", *dataset, err.Error())
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
		MutateRate:  1,
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
		100, *grpcPort, *clusterPort, *metricsPort,
		strings.Split(*clusterBoostrap, ","),
		//rand.New(rand.NewSource(7)))
		rand.New(rand.NewSource(time.Now().Unix())),
		mlp.NewRandMLP)

	// Configure  extra pool settings
	pool.WebPort = *webPort
	pool.WebPath = *webPath
	pool.Delegate = mlp.DelegateImpl{}
	pool.KeepBest = false
	pool.NMigrate = pool.PopSize / 10
	pool.SortPrecission = 100
	pool.SortFunc = mlp.SortByFitnessAndNeurons
	pool.CrossRate = 0.3
	pool.MutRate = 0.3
	pool.MaxEvaluations = 2000
	pool.ExtraOperators = []eaopt.ExtraOperator{
		eaopt.ExtraOperator{Operator: mlp.AddNeuron, Probability: 0.3},
		eaopt.ExtraOperator{Operator: mlp.RemoveNeuron, Probability: 0.15},
		eaopt.ExtraOperator{Operator: mlp.SubstituteNeuron, Probability: 0.15},
		eaopt.ExtraOperator{Operator: mlp.Train, Probability: 0.5},
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
