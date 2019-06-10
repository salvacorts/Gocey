package main

import (
	"io/ioutil"
	"log"
	"os"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common"
)

func main() {
	log.SetOutput(os.Stdout)

	filename := "../../datasets/iris.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatalf("Cannot open %s. Error: %s", filename, err.Error())
	}

	_, scores := common.TrainMLP(string(fileContent))

	log.Printf("Scores: %v\n", scores)
}
