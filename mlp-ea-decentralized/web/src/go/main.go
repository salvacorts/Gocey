package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"syscall/js"
	"time"

	"github.com/dennwc/dom/net/ws"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/ga"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"

	"github.com/sirupsen/logrus"
)

func main() {
	log.SetOutput(WebLogger{})

	// Ask for served grpc port
	resp, err := http.Get("/grpcPortWS")
	if err != nil {
		log.Fatalf("Could not ask for grpcPort. %s", err.Error())
	}
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Could not read Body. %s", err.Error())
	}

	client := ga.Client{
		ID:       "clientWasm",
		Log:      logrus.New(),
		Delegate: mlp.DelegateImpl{},

		// Use WebSockets
		CustomDialer: func(s string, dt time.Duration) (net.Conn, error) {
			return ws.Dial(s)
		},
	}

	logger := WebLogger{}
	logrus.SetOutput(logger)
	logrus.SetLevel(logrus.ErrorLevel)
	client.Log.SetOutput(logger)
	client.Log.SetLevel(logrus.InfoLevel)
	client.Log.SetFormatter(&logrus.JSONFormatter{})

	host := js.Global().Call("GetHostName")
	client.Log.Infof("Hostname: %s", host.String())
	client.ServerAddr = fmt.Sprintf("ws://%s:%s", host.String(), string(body))

	err = client.Start()
	if err != nil {
		client.Log.Fatalf("Got error from client: %s", err.Error())
	}
}
