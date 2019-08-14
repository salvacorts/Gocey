package main

import (
	"flag"
	"os"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/ga"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
	"google.golang.org/grpc/grpclog"

	"github.com/sirupsen/logrus"
)

var (
	serverAddr = flag.String("server", "127.0.0.1:3117", "Server address in format addr:port.")
)

func main() {
	flag.Parse()

	client := ga.Client{
		ServerAddr: *serverAddr,
		ID:         "clientNative",
		Log:        logrus.New(),
		Delegate:   mlp.DelegateImpl{},
	}

	logrus.SetLevel(logrus.ErrorLevel)
	client.Log.SetLevel(logrus.InfoLevel)
	grpclog.SetLoggerV2(grpclog.NewLoggerV2(os.Stdout, os.Stdout, os.Stdout))

	err := client.Start()
	if err != nil {
		logrus.Fatalf("Got error from client: %s", err.Error())
	}
}
