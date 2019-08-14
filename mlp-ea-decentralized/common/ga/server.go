//+build !js

package ga

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/golang/protobuf/ptypes/empty"
	"github.com/salvacorts/eaopt"
	"github.com/sirupsen/logrus"
)

// Server implements DistributedEA service
type Server struct {
	Log    *logrus.Logger
	Input  chan eaopt.Individual
	Output chan eaopt.Individual
	Pool   *PoolModel

	clients int
}

// GetProblemDescription returs the configuration for the problem execution to the client
func (s *Server) GetProblemDescription(ctx context.Context, in *empty.Empty) (*ProblemDescription, error) {
	description := &ProblemDescription{
		ClientID: fmt.Sprintf("client%d", s.clients),
		Payload:  s.Pool.Delegate.SerializeProblemDescription(),
	}

	s.clients++

	return description, nil
}

// GetStats returs the statistics of the overall execution on the algorithm
func (s *Server) GetStats(ctx context.Context, in *empty.Empty) (*Stats, error) {
	stats := &Stats{
		Evaluations: int64(s.Pool.GetTotalEvaluations()),
		BestFitness: s.Pool.BestSolution.Fitness,
		AvgFitness:  s.Pool.GetAverageFitness(),
	}

	return stats, nil
}

// BorrowIndividual implements DistributedEA service
func (s *Server) BorrowIndividual(ctx context.Context, e *empty.Empty) (*Individual, error) {
	// Read from channel and give it to client
	o1, ok := <-s.Input
	if !ok {
		s.Log.Debugln("Server Input channel was already closed")
		return nil, errors.New("Input channel in server is closed")
	}

	msg := &Individual{
		IndividualID: o1.ID,
		Evaluated:    false,
		Fitness:      o1.Fitness,

		Genome: s.Pool.Delegate.SerializeGenome(o1.Genome),
	}

	return msg, nil
}

// ReturnIndividual implements DistributedEA service
func (s *Server) ReturnIndividual(ctx context.Context, msg *Individual) (*empty.Empty, error) {
	opLogger := s.Log.WithFields(logrus.Fields{
		"client":    msg.ClientID,
		"Evaluated": msg.Evaluated,
		"Fitness":   msg.Fitness,
	})

	// Should never get inside since returned individuals should be evaluated
	if !msg.Evaluated {
		opLogger.Errorln("The MLP return from the client is not evaluated")

		s.Input <- eaopt.Individual{
			Genome:    s.Pool.Delegate.DeserializeGenome(msg.Genome),
			Fitness:   math.Inf(1),
			Evaluated: false,
			ID:        msg.IndividualID,
		}

		opLogger.Debugln("Appended MLP back to the input channel")
	}

	// Put mlp into output channel
	s.Output <- eaopt.Individual{
		Genome:    s.Pool.Delegate.DeserializeGenome(msg.Genome),
		Fitness:   msg.Fitness,
		Evaluated: true,
		ID:        msg.IndividualID,
	}

	opLogger.Debugln("Received Evaluated eaopt.Individual -> output chan")

	return &empty.Empty{}, nil
}
