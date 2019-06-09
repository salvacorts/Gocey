package main

import (
	"io/ioutil"
	"log"
	"net/http"
	"syscall/js"

	common "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common"
)

func trainMLPWrapper(this js.Value, args []js.Value) interface{} {
	go func() {
		resp, err := http.Get("http://127.0.0.1:8081/datasets/glass.csv")
		if err != nil {
			log.Fatalf("Cannot get dataset. Error: %s", err.Error())
		}

		defer resp.Body.Close()

		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			log.Fatalf("Cannot read dataset body. Error: %s", err.Error())
		}

		common.TrainMLP((string(body)))
	}()

	return nil
}

func main() {
	// TODO: Make logs go to website
	trainMLPcb := js.FuncOf(trainMLPWrapper)
	js.Global().Set("trainMLP", trainMLPcb)

	select {}
}
