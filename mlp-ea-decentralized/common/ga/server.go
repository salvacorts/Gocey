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

// MigrateIndividuals implements DistributedEA service
func (s *Server) MigrateIndividuals(ctx context.Context, batch *IndividualsBatch) (*empty.Empty, error) {
	candidates := make([]eaopt.Individual, len(batch.Individuals))

	scores := make([]float64, 0, len(batch.Individuals))
	for _, in := range batch.Individuals {
		scores = append(scores, in.Fitness)
	}

	s.Log.Infof("Got migration of %d individuals: %v", len(batch.Individuals), scores)
	s.Pool.metricsServer.IncomingMigrationsCount.Inc()

	for i, indiv := range batch.Individuals {
		candidates[i] = eaopt.Individual{
			ID:        indiv.IndividualID,
			Evaluated: indiv.Evaluated,
			Fitness:   indiv.Fitness,
			Genome:    s.Pool.Delegate.DeserializeGenome(indiv.Genome),
		}
	}

	// Sort the population and the candidates
	population := s.Pool.GetPopulationSnapshot()
	population = s.Pool.SortFunc(population, s.Pool.SortPrecission)
	candidates = s.Pool.SortFunc(candidates, s.Pool.SortPrecission)

	//toRemove := make([]eaopt.Individual, len(candidates))
	toAppend := make([]eaopt.Individual, len(candidates))

	// Check if the worst from the population are
	// actually better than the bests from the candidates
	i, j, k := len(population)-len(candidates), 0, 0
	for k < s.Pool.NMigrate {
		if population[i].Fitness < candidates[j].Fitness {
			toAppend[k] = population[i]
			i++
		} else {
			toAppend[k] = candidates[j]
			j++
		}

		k++
	}

	// Append the best candidates
	for a := range toAppend {
		s.Pool.population.Add(toAppend[a])
	}

	if toAppend[0].Fitness < s.Pool.BestSolution.Fitness {
		s.Log.Infof("Form migration: New best solution with fitness: %f", toAppend[0].Fitness)
		s.Pool.BestSolution = toAppend[0]
	}

	// Remove the worst ones from the population that were
	// not better than the candidates
	for a := i; a < len(population); a++ {
		s.Pool.population.Remove(population[a])
	}

	return &empty.Empty{}, nil
}
