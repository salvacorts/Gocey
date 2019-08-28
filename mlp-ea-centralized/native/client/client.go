package main

import (
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common/ga"

	"github.com/sirupsen/logrus"
)

func main() {
	client := ga.MLPClient{
		ServerAddr: "127.0.0.1:3117",
		ID:         "clientNative",
		Log:        logrus.New(),
	}

	logrus.SetLevel(logrus.ErrorLevel)
	client.Log.SetLevel(logrus.InfoLevel)

	err := client.Start()
	if err != nil {
		logrus.Fatalf("Got error from client: %s", err.Error())
	}
}
