package common

import (
	"math"
	"math/rand"
	"net"
	"reflect"
	"runtime"
	"sync"

	"github.com/sirupsen/logrus"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-centralized/common/mlp"

	"github.com/salvacorts/eaopt"
	"google.golang.org/grpc"
)

type indivInfo struct {
	Generation int
	Individual eaopt.Individual
}

// Pipe represents a function that applies an operator to a Genome that
// comes from the first channel and goes through the second channel
type Pipe = func(in, out chan eaopt.Individual)

// PipedPoolModel is a evolutionary algorithm model that uses channels to apply
// genetic operators over a population
type PipedPoolModel struct {
	Rnd            *rand.Rand
	KeepBest       bool
	SortFunction   func(eaopt.Individuals)
	ExtraOperators []eaopt.ExtraOperator
	EarlyStop      func(*PipedPoolModel) bool

	// Callbacks
	BestCallback       func(*PipedPoolModel)
	GenerationCallback func(*PipedPoolModel)

	CrossRate float64
	MutRate   float64

	PopSize       int
	MaxGeneration int
	Generation    int
	BestSolution  eaopt.Individual

	population map[string]*indivInfo
	pipeline   []Pipe
	stop       chan bool
	wg         sync.WaitGroup
	grpcServer *grpc.Server

	// Stats
	fitAvg float64
}

// MakePool Creates a new pool with default configuration
func MakePool() PipedPoolModel {
	return PipedPoolModel{
		Generation: 0,
		stop:       make(chan bool),
		population: make(map[string]*indivInfo),
		BestSolution: eaopt.Individual{
			Fitness: math.Inf(1),
		},
	}
}

// FitAvg returns the average fitness of the population
func (mod *PipedPoolModel) FitAvg() float64 {
	return mod.fitAvg
}

// Init initialized the model by creating a pipeline of channels with the different
// genetic operators
func (mod *PipedPoolModel) init() {
	// Append Cross and Mutate
	mod.pipeline = append(mod.pipeline, mod.crossover, mod.mutate)

	// Append the rest of extra operators
	for _, operator := range mod.ExtraOperators {
		mod.pipeline = append(mod.pipeline, func(in, out chan eaopt.Individual) {
			defer mod.wg.Done()
			defer close(out)

			for {
				select {
				default:
					o1, ok := <-in
					if !ok {
						Log.Debugln("in was already closed")
						return
					}

					if mod.Rnd.Float64() < operator.Probability {
						o1.ApplyExtraOperator(operator, mod.Rnd)
					}

					out <- o1
				case <-mod.stop:
					return
				}
			}
		})
	}

	// Append Evaluate and Control ops
	mod.pipeline = append(mod.pipeline, mod.evaluate, mod.generationControl)

	// Create initial population
	for i := 0; i < mod.PopSize; i++ {
		indi := eaopt.NewIndividual(NewRandMLP(mod.Rnd), mod.Rnd)
		mod.population[indi.ID] = &indivInfo{0, indi}
	}
}

// Minimize runs the algorithm by connecting the pipes
func (mod *PipedPoolModel) Minimize() {
	mod.init()

	mod.wg.Add(len(mod.pipeline))

	// Connect the pipeline creating a circular pipeline
	start := make(chan eaopt.Individual, mod.PopSize)
	previous := start

	Log.Debugf("Start: %v", start)

	for i := 0; i < len(mod.pipeline)-1; i++ {
		current := make(chan eaopt.Individual, mod.PopSize)

		Log.Debugf("%v -> %s -> %v", previous, getFunctionName(mod.pipeline[i]), current)

		go mod.pipeline[i](previous, current)

		previous = current
	}

	Log.Debugf("%v -> %s -> %v", previous, getFunctionName(mod.pipeline[len(mod.pipeline)-1]), start)

	// Connect the last pipe to the first one
	go mod.pipeline[len(mod.pipeline)-1](previous, start)

	// Put all individuals of the population in the first channel
	for _, indiInfo := range mod.population {
		start <- indiInfo.Individual
	}

	mod.wg.Wait()
}

func (mod *PipedPoolModel) crossover(in, out chan eaopt.Individual) {
	defer mod.wg.Done()
	defer close(out)

	for {
		select {
		default:
			p1, ok1 := <-in
			p2, ok2 := <-in
			if !ok1 || !ok2 {
				Log.Debugln("in was already closed")
				return
			}

			if mod.Rnd.Float64() < mod.CrossRate {

				// Create offsprings
				o1, o2 := p1.Clone(mod.Rnd), p2.Clone(mod.Rnd)
				o1.Crossover(o2, mod.Rnd)

				// Keep thre ID of the parents
				o1.ID, o2.ID = p1.ID, p2.ID

				out <- o1
				out <- o2
			} else {
				out <- p1
				out <- p2
			}
		case <-mod.stop:
			return
		}
	}
}

func (mod *PipedPoolModel) mutate(in, out chan eaopt.Individual) {
	defer mod.wg.Done()
	defer close(out)

	for {
		select {
		default:
			o1, ok := <-in
			if !ok {
				Log.Debugln("in was already closed")
				return
			}

			if mod.Rnd.Float64() < mod.MutRate {
				o1.Mutate(mod.Rnd)
			}

			out <- o1
		case <-mod.stop:
			return
		}
	}
}

func (mod *PipedPoolModel) generationControl(in, out chan eaopt.Individual) {
	defer mod.wg.Done()
	defer close(out)

	for {
		select {
		default:
			o1, ok := <-in
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
			}

			// TODO: Do this better
			info := mod.population[o1.ID]
			info.Generation++

			if info.Generation > mod.Generation+1 {
				mod.Generation++

				if mod.KeepBest {
					// Find the worst solution
					var worstID string
					worst := math.Inf(-1)

					for key, value := range mod.population {
						if value.Individual.Fitness > worst {
							worst = value.Individual.Fitness
							worstID = key
						}
					}

					if Log.Level == logrus.DebugLevel {
						var keys []string
						for key := range mod.population {
							keys = append(keys, key)
						}

						Log.Debugf("Keys in Population: %v", keys)
					}

					// Replace the worst solution by the best one so far
					mod.population[worstID] = &indivInfo{
						Generation: mod.population[worstID].Generation,
						Individual: mod.BestSolution,
					}
				}

				if mod.GenerationCallback != nil {
					mod.GenerationCallback(mod)
				}

				if mod.Generation >= mod.MaxGeneration || mod.EarlyStop(mod) {
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
			}

			out <- o1
		case <-mod.stop:
			return
		}
	}
}

func (mod *PipedPoolModel) evaluate(in, out chan eaopt.Individual) {
	defer mod.wg.Done()
	defer close(out)

	mlpServer := &MLPServer{
		Input:  in,
		Output: out,
		Log:    Log,
	}

	// TODO: Make it use UDP
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
