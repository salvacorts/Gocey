package main

import (
	"io/ioutil"
	"log"
	"net/http"

	"github.com/dennwc/dom"
	common "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common"
)

func main() {
	log.SetOutput(WebLogger{})

	resp, err := http.Get("/datasets/glass.csv")
	if err != nil {
		log.Fatalf("Cannot get dataset. Error: %s", err.Error())
	}

	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Cannot read dataset body. Error: %s", err.Error())
	}

	common.TrainMLP((string(body)))

	dom.Loop()
}
