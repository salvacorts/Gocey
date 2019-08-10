package client

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"

	"github.com/golang/protobuf/ptypes/empty"
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common/mlp"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

// MLPClient is a client to communicate with the GRPC server
type MLPClient struct {
	ServerAddr   string
	Log          *logrus.Logger
	ID           string
	CustomDialer func(s string, dt time.Duration) (net.Conn, error)
}

// Start starts the client on a while loop borrowing, evaluating and returning individuals
func (c *MLPClient) Start() error {
	var conn *grpc.ClientConn
	var err error

	c.Log.Infof("Dialing %s", c.ServerAddr)

	// For wasm use WebSockets
	if c.CustomDialer != nil {
		conn, err = grpc.Dial(
			c.ServerAddr, grpc.WithDialer(c.CustomDialer), grpc.WithInsecure())

		c.Log.Debugf("Using custom dialer\n")
	} else {
		conn, err = grpc.Dial(c.ServerAddr, grpc.WithInsecure())
	}

	if err != nil {
		c.Log.Fatalf("Cannot Dial. Error: %s", err.Error())
		return err
	}

	defer conn.Close()

	client := mlp.NewDistributedEAClient(conn)

	desc, err := client.GetProblemDescription(context.Background(), &empty.Empty{})
	if err != nil {
		c.Log.Fatalf("Cannot get problem description from server. Error: %s", err.Error())
	}

	fmt.Printf(desc.TrainDataset)

	mlp.Config.Epochs = int(desc.Epochs)
	mlp.Config.Folds = int(desc.Folds)
	mlp.Config.Classes = desc.Classes
	mlp.Config.TrainingSet, err, _ = utils.LoadPatternsFromCSV(desc.TrainDataset)
	if err != nil {
		c.Log.Fatalf("Cannot parse CSV patterns from got from server. Error: %s", err.Error())
	}

	c.Log.Debugf("Patterns length: %d", len(mlp.Config.TrainingSet))

	localEvaluations := 0
	start := time.Now()

	for {
		msg, err := client.BorrowIndividual(context.Background(), &empty.Empty{})
		if err != nil {
			c.Log.Fatalf("Cannot borrow individual from server. Error: %s", err.Error())
			break
		}

		nn := (*mlp.MLP)(msg.Mlp)
		score, err := nn.Evaluate()
		if err != nil {
			c.Log.Fatalf("Error evaluating MLP. Error: %s", err.Error())
			break
		}

		c.Log.Infof("Got score: %f", score)

		out := &mlp.MLPMsg{
			Mlp:          (*mlp.MultiLayerNetwork)(nn),
			IndividualID: msg.IndividualID,
			ClientID:     c.ID,
			Fitness:      score,
			Evaluated:    true,
		}

		_, err = client.ReturnIndividual(context.Background(), out)
		if err != nil {
			c.Log.Fatalf("Cannot return back individual to server. Error: %s", err.Error())
			break
		}

		localEvaluations++

		if localEvaluations%1000 == 0 {
			c.Log.Infof("Throughput: %d evaluations / second", localEvaluations/int(time.Since(start).Seconds()))
		}
	}

	return nil
}
