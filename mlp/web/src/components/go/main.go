package main

import (
	"syscall/js"

	common "github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common"
)

func trainMLPWrapper(this js.Value, args []js.Value) interface{} {
	common.TrainMLP("datasets/glass.csv")
	return nil
}

func main() {
	// TODO: Make logs go to website
	trainMLPcb := js.FuncOf(trainMLPWrapper)
	js.Global().Set("trainMLP", trainMLPcb)

	select {}
}
