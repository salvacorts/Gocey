package main

import (
	"os"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common/ga/client"
	"google.golang.org/grpc/grpclog"

	"github.com/sirupsen/logrus"
)

func main() {
	client := client.MLPClient{
		ServerAddr: "127.0.0.1:3117",
		ID:         "clientNative",
		Log:        logrus.New(),
	}

	logrus.SetLevel(logrus.ErrorLevel)
	client.Log.SetLevel(logrus.InfoLevel)
	grpclog.SetLoggerV2(grpclog.NewLoggerV2(os.Stdout, os.Stdout, os.Stdout))

	err := client.Start()
	if err != nil {
		logrus.Fatalf("Got error from client: %s", err.Error())
	}
}
