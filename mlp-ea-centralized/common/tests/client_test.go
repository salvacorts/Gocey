package tests

import (
	"testing"

	"github.com/sirupsen/logrus"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common"
)

func TestClient(t *testing.T) {
	client := common.MLPClient{
		ServerAddr: "127.0.0.1:3117",
		ID:         "client1",
		Log:        logrus.New(),
	}

	client.Log.SetLevel(logrus.DebugLevel)

	err := client.Start()
	if err != nil {
		t.Errorf("Got error from client: %s", err.Error())
	}
}
