package main

import (
	"log"
	"os"

	common "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common"
)

func main() {
	log.SetOutput(os.Stdout)
	common.TrainMLP("../../datasets/glass.csv")
}
