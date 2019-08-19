package ga

import (
	"context"
	"net"
	"time"

	//"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"

	"github.com/golang/protobuf/ptypes/empty"
	//"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

// Client is a client to communicate with the GRPC server
type Client struct {
	ServerAddr   string
	Log          *logrus.Logger
	ID           string
	CustomDialer func(s string, dt time.Duration) (net.Conn, error)
	Delegate     ServiceDelegate

	finalize bool
}

// StatsFetcher is a thread to ask server about stats {best_fitness, best_neurons, total_evaluations_overall}
func (c *Client) StatsFetcher(client *DistributedEAClient) {
	for !c.finalize {
		time.Sleep(30 * time.Second)

		stats, err := (*client).GetStats(context.Background(), &empty.Empty{})
		if err != nil {
			c.Log.Errorf("Could not get stats from server")
		}

		c.Log.WithFields(logrus.Fields{
			"level":       "info",
			"Scope":       "remote",
			"Evaluations": stats.Evaluations,
			"BestFitness": stats.BestFitness,
			"AvgFitness":  stats.AvgFitness,
		}).Infof("Got Stats from server")
	}
}

// Start starts the client on a while loop borrowing, evaluating and returning individuals
func (c *Client) Start() error {
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

	client := NewDistributedEAClient(conn)

	c.finalize = false
	go c.StatsFetcher(&client)

	desc, err := client.GetProblemDescription(context.Background(), &empty.Empty{})
	if err != nil {
		c.Log.Fatalf("Cannot get problem description from server. Error: %s", err.Error())
	}

	c.Delegate.DeserializeProblemDescription(desc.Payload)

	localEvaluations := 0
	start := time.Now()

	for {
		msg, err := client.BorrowIndividual(context.Background(), &empty.Empty{})
		if err != nil {
			c.Log.Fatalf("Cannot borrow individual from server. Error: %s", err.Error())
			break
		}

		genome := c.Delegate.DeserializeGenome(msg.Genome)
		score, err := genome.Evaluate()
		if err != nil {
			c.Log.Fatalf("Error evaluating Genome. Error: %s", err.Error())
			break
		}

		c.Log.WithFields(logrus.Fields{
			"level":       "info",
			"Scope":       "local",
			"Evaluations": localEvaluations,
			"Fitness":     score,
		}).Infof("Got score: %f", score)

		out := &Individual{
			IndividualID: msg.IndividualID,
			Evaluated:    true,
			Fitness:      score,

			// TODO: Do not serialize again since you have not modified it
			// 			pass msg.Payload as argument
			Genome: c.Delegate.SerializeGenome(genome),
		}

		_, err = client.ReturnIndividual(context.Background(), out)
		if err != nil {
			c.Log.Fatalf("Cannot return back individual to server. Error: %s", err.Error())
			break
		}

		localEvaluations++

		if localEvaluations%100 == 0 {
			c.Log.Infof("Throughput: %d evaluations / second", localEvaluations/int(time.Since(start).Seconds()))
		}
	}

	c.finalize = true

	return nil
}
