package main

import (
	"log"
	"net"
	"time"

	"github.com/dennwc/dom/net/ws"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/ga"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"

	"github.com/sirupsen/logrus"
)

func main() {
	log.SetOutput(WebLogger{})

	logger := WebLogger{}

	client := ga.Client{
		ServerAddr: "ws://127.0.0.1:2019",
		ID:         "clientWasm",
		Log:        logrus.New(),
		Delegate:   mlp.DelegateImpl{},

		// Use WebSockets
		CustomDialer: func(s string, dt time.Duration) (net.Conn, error) {
			return ws.Dial(s)
		},
	}

	logrus.SetOutput(logger)
	logrus.SetLevel(logrus.ErrorLevel)
	client.Log.SetOutput(logger)
	client.Log.SetLevel(logrus.InfoLevel)
	client.Log.SetFormatter(&logrus.JSONFormatter{})

	err := client.Start()
	if err != nil {
		client.Log.Fatalf("Got error from client: %s", err.Error())
	}
}
