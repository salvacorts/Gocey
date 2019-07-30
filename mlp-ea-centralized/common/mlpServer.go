package common

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"

	"github.com/sirupsen/logrus"

	"github.com/salvacorts/eaopt"

	"github.com/golang/protobuf/ptypes/empty"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common/mlp"
)

// MLPServer implements DistributedEA service
type MLPServer struct {
	Log    *logrus.Logger
	Input  chan eaopt.Individual
	Output chan eaopt.Individual

	clients int // TODO: Timeout clients
}

// GetProblemDescription returs the configuration for the problem execution to the client
func (s *MLPServer) GetProblemDescription(ctx context.Context, in *empty.Empty) (*mlp.ProblemDescription, error) {
	description := &mlp.ProblemDescription{
		ClientID:     fmt.Sprintf("client%d", s.clients),
		Epochs:       int64(Config.Epochs),
		TrainDataset: utils.PatternsToCSV(Config.TrainingSet),
		Classes:      Config.Classes,
	}

	s.clients++

	return description, nil
}

// BorrowIndividual implements DistributedEA service
func (s *MLPServer) BorrowIndividual(ctx context.Context, e *empty.Empty) (*mlp.MLPMsg, error) {
	// Read from channel and give it to client
	o1, ok := <-s.Input
	if !ok {
		s.Log.Debugln("Server Input channel was already closed")
		return nil, errors.New("Input channel in server is closed")
	}

	msg := &mlp.MLPMsg{
		Mlp:          (*mlp.MultiLayerNetwork)(o1.Genome.(*MLP)),
		IndividualID: o1.ID,
		Evaluated:    false,
		Fitness:      o1.Fitness,
	}

	return msg, nil
}

// ReturnIndividual implements DistributedEA service
func (s *MLPServer) ReturnIndividual(ctx context.Context, msg *mlp.MLPMsg) (*empty.Empty, error) {
	opLogger := s.Log.WithFields(logrus.Fields{
		"client":    msg.ClientID,
		"Evaluated": msg.Evaluated,
		"Fitness":   msg.Fitness,
	})

	if !msg.Evaluated {
		opLogger.Errorln("The MLP return from the client is not evaluated")

		s.Input <- eaopt.Individual{
			Genome:    (*MLP)(msg.Mlp),
			Fitness:   math.Inf(1),
			Evaluated: false,
			ID:        msg.IndividualID,
		}

		opLogger.Debugln("Appended MLP back to the input channel")
	}

	// Put mlp into output channel
	s.Output <- eaopt.Individual{
		Genome:    (*MLP)(msg.Mlp),
		Fitness:   msg.Fitness,
		Evaluated: true,
		ID:        msg.IndividualID,
	}

	opLogger.Debugln("Received Evaluated Individual -> output chan")

	return &empty.Empty{}, nil
}
