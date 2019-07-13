package main

import (
	"io/ioutil"
	"log"
	"net/http"

	"github.com/dennwc/dom"
	ga "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea/common"
	"github.com/sirupsen/logrus"
)

func main() {
	log.SetOutput(WebLogger{})

	logger := WebLogger{}

	logrus.SetOutput(logger)
	logrus.SetLevel(logrus.ErrorLevel)
	ga.Log.SetOutput(logger)
	ga.Log.SetLevel(logrus.InfoLevel)
	ga.Log.SetFormatter(&logrus.JSONFormatter{})

	resp, err := http.Get("/datasets/glass.csv")
	if err != nil {
		log.Fatalf("Cannot get dataset. Error: %s", err.Error())
	}

	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Cannot read dataset body. Error: %s", err.Error())
	}

	_, score, err := ga.TrainMLP(string(body))
	if err != nil {
		log.Printf("Error Training MLP: %s", err.Error())
	}

	log.Printf("Scores: %v\n", score)

	dom.Loop()
}
