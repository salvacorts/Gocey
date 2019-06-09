package main

import (
	"io/ioutil"
	"log"
	"os"

	common "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common"
)

func main() {
	log.SetOutput(os.Stdout)

	filename := "../../datasets/glass.csv"
	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatalf("Cannot open %s. Error: %s", filename, err.Error())
	}

	common.TrainMLP(string(fileContent))
}
