package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	ga "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea/common"
	"github.com/sirupsen/logrus"
)

func main() {
	logrus.SetOutput(os.Stdout)
	logrus.SetLevel(logrus.ErrorLevel)
	ga.Log.SetOutput(os.Stdout)
	ga.Log.SetLevel(logrus.InfoLevel)

	filename := "../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatalf("Cannot open %s. Error: %s", filename, err.Error())
	}

	_, score, err := ga.TrainMLP(string(fileContent))
	if err != nil {
		fmt.Printf("Error Training MLP: %s", err.Error())
	}

	fmt.Printf("Score: %v\n", score)
}
