package common

import (
	"fmt"
	"math"
	"math/rand"
	"net"
	"reflect"
	"runtime"
	"sync"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common/mlp"

	"github.com/salvacorts/eaopt"
	"google.golang.org/grpc"
)

type semaphore chan int

func (sem semaphore) Acquire(n int) {
	for i := 0; i < n; i++ {
		<-sem
	}
}

func (sem semaphore) Release(n int) {
	for i := 0; i < n; i++ {
		sem <- 1
	}
}

func removeIndividual(in []eaopt.Individual, i int) []eaopt.Individual {
	return append(in[:i], in[i+1:]...)
}

// Pipe represents a function that applies an operator to a Genome that
// comes from the first channel and goes through the second channel
type Pipe = func(in, out chan eaopt.Individual)

// PipedPoolModel is a evolutionary algorithm model that uses channels to apply
// genetic operators over a population
type PipedPoolModel struct {
	Rnd            *rand.Rand
	KeepBest       bool
	SortFunction   func([]eaopt.Individual)
	ExtraOperators []eaopt.ExtraOperator
	EarlyStop      func(*PipedPoolModel) bool

	// Callbacks
	BestCallback       func(*PipedPoolModel)
	GenerationCallback func(*PipedPoolModel)

	CrossRate float64
	MutRate   float64

	PopSize        int
	MaxEvaluations int
	BestSolution   eaopt.Individual

	currentPopSize int
	population     []eaopt.Individual
	popSemaphore   semaphore
	pipeline       []Pipe

	evaluations       int
	stop              chan bool
	wg                sync.WaitGroup
	grpcServer        *grpc.Server
	evaluationChannel chan eaopt.Individual
	evaluatedChannel  chan eaopt.Individual

	// Stats
	fitAvg float64
}

// MakePool Creates a new pool with default configuration
func MakePool(popSize int, rnd *rand.Rand) PipedPoolModel {
	pool := PipedPoolModel{
		Rnd:               rnd,
		PopSize:           popSize,
		population:        make([]eaopt.Individual, popSize),
		popSemaphore:      make(semaphore, popSize),
		evaluations:       0,
		stop:              make(chan bool),
		evaluationChannel: make(chan eaopt.Individual, popSize),
		evaluatedChannel:  make(chan eaopt.Individual, popSize),
		BestSolution: eaopt.Individual{
			Fitness: math.Inf(1),
		},
	}

	for i := 0; i < pool.PopSize; i++ {
		pool.population[i] =
			eaopt.NewIndividual(NewRandMLP(pool.Rnd), pool.Rnd)
	}

	pool.currentPopSize = pool.PopSize

	return pool
}

// FitAvg returns the average fitness of the population
func (mod *PipedPoolModel) FitAvg() float64 {
	return mod.fitAvg
}

// Minimize runs the algorithm by connecting the pipes
func (mod *PipedPoolModel) Minimize() {
	mod.wg.Add(2)
	var offsprings []eaopt.Individual

	mod.popSemaphore.Release(popSize)

	fmt.Printf("\n\nPopulation: ")
	for i := range mod.population {
		fmt.Printf("%s, ", mod.population[i].ID)
	}
	fmt.Printf("\n")

	go mod.handleEvaluate()
	go mod.handleEvaluated()

	for mod.evaluations < mod.MaxEvaluations {
		// take here randomly from population
		offsprings = mod.selection(4)
		offsprings = mod.crossover(offsprings)
		offsprings = mod.mutate(offsprings)

		// Apply extra operators
		for i := range offsprings {
			for _, op := range mod.ExtraOperators {
				if mod.Rnd.Float64() <= op.Probability {
					offsprings[i].Genome = op.Operator(offsprings[i].Genome, mod.Rnd)
					offsprings[i].Evaluated = false
				}
			}
		}

		// Append non evaluated offspings to the channel and
		// evaluated one again to the population
		for i := range offsprings {
			if !offsprings[i].Evaluated {
				mod.evaluationChannel <- offsprings[i]
			} else {
				mod.population = append(mod.population, offsprings[i])
				mod.popSemaphore.Release(1)
			}
		}
	}

	mod.wg.Wait()
}

// Binary torunament
// TODO: use semaphore here
func (mod *PipedPoolModel) selection(n int) []eaopt.Individual {
	offsprings := make([]eaopt.Individual, n)

	mod.popSemaphore.Acquire(n)
	if len(mod.population) < n {
		Log.Fatal("Not enought eaopt.Individuals on the population to select")
	}

	for i := range offsprings {
		rnd1 := mod.Rnd.Intn(len(mod.population))
		rnd2 := mod.Rnd.Intn(len(mod.population))

		// Make them differrent.
		for rnd1 == rnd2 {
			rnd2 = mod.Rnd.Intn(len(mod.population))
		}

		if mod.population[rnd1].Fitness < mod.population[rnd2].Fitness {
			offsprings[i] = mod.population[rnd1]
			mod.population = removeIndividual(mod.population, rnd1)
		} else {
			offsprings[i] = mod.population[rnd2]
			mod.population = removeIndividual(mod.population, rnd2)
		}
	}

	return offsprings
}

// TODO: Get rid of odd arrays here
func (mod *PipedPoolModel) crossover(in []eaopt.Individual) []eaopt.Individual {
	new := make([]eaopt.Individual, len(in))

	// Copy all the offsprings
	for i := range new {
		new[i] = in[i].Clone(mod.Rnd)
	}

	// Sort clones
	SortByFitnessAndNeurons(new)

	// With the offspring first half, replace the second half
	for i := 0; i < len(new)/2; i += 2 {
		o1, o2 := new[i].Clone(mod.Rnd), new[i+1].Clone(mod.Rnd)
		o1.Genome.Crossover(o2.Genome, mod.Rnd)
		o1.Evaluated = false
		o2.Evaluated = false

		new[len(new)-1-i] = o1
		new[len(new)-1-(i+1)] = o2
	}

	return new
}

func (mod *PipedPoolModel) mutate(in []eaopt.Individual) []eaopt.Individual {
	for i := range in {
		if mod.Rnd.Float64() < mod.MutRate {
			in[i].Evaluated = false
			in[i].Genome.Mutate(mod.Rnd)
		}
	}

	return in
}

func (mod *PipedPoolModel) handleEvaluated() {
	defer mod.wg.Done()

	for {
		select {
		default:
			o1, ok := <-mod.evaluatedChannel
			if !ok {
				Log.Debugln("in was already closed")
				return
			}

			// Calculate average fitness here
			mod.fitAvg += o1.Fitness
			mod.fitAvg /= 2

			if o1.Fitness < mod.BestSolution.Fitness {
				mod.BestSolution = o1.Clone(mod.Rnd)

				if mod.BestCallback != nil {
					mod.BestCallback(mod)
				}

				fmt.Printf("\n\nPopulation [%d]: ", len(mod.population))
				for i := range mod.population {
					fmt.Printf("%s, ", mod.population[i].ID)
				}
				fmt.Printf("\n")
			}

			mod.population = append(mod.population, o1)
			mod.popSemaphore.Release(1)
			mod.evaluations++

			// if mod.evaluations%100 == 0 {
			// 	fmt.Printf("\n\nPopulation [%d]: ", len(mod.population))
			// 	for i := range mod.population {
			// 		fmt.Printf("%s, ", mod.population[i].ID)
			// 	}
			// 	fmt.Printf("\n")
			// }

			if mod.evaluations >= mod.MaxEvaluations || mod.EarlyStop(mod) {
				select {
				case <-mod.stop:
					Log.Debugln("mod.stop was already closed")
				default:
					if mod.stop != nil {
						mod.grpcServer.Stop()
						close(mod.stop)
					}
				}
			}

		case <-mod.stop:
			return
		}
	}
}

func (mod *PipedPoolModel) handleEvaluate() {
	defer mod.wg.Done()

	mlpServer := &MLPServer{
		Input:  mod.evaluationChannel,
		Output: mod.evaluatedChannel,
		Log:    Log,
	}

	listener, err := net.Listen("tcp", ":3117")
	if err != nil {
		Log.Fatalf("Failed to listen: %v", err)
	}

	// Start listening on server
	mod.grpcServer = grpc.NewServer()
	mlp.RegisterDistributedEAServer(mod.grpcServer, mlpServer)
	mod.grpcServer.Serve(listener)

	// Wait until stop is received
	select {
	case <-mod.stop:
	}
}

func getFunctionName(i interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
}
