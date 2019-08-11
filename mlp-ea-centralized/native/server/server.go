package main

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common/ga"
	"google.golang.org/grpc/grpclog"

	"github.com/sirupsen/logrus"
)

func main() {
	filename := "../../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		logrus.Fatalf("Cannot open %s. Error: %s", filename, err.Error())
	}

	logrus.SetOutput(os.Stdout)
	logrus.SetLevel(logrus.ErrorLevel)
	ga.Log.SetOutput(os.Stdout)
	ga.Log.SetLevel(logrus.InfoLevel)
	grpclog.SetLoggerV2(grpclog.NewLoggerV2(ga.Log.Out, ga.Log.Out, ga.Log.Out))

	_, score, err := ga.TrainMLP(string(fileContent))
	if err != nil {
		logrus.Fatalf("Error Training MLP: %s", err.Error())
	}

	fmt.Printf("Got Error: %f\n", score)
}