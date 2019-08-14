//+build !js

package ga

import (
	"fmt"
	"math"
	"math/rand"
	"net"
	"reflect"
	"runtime"
	"sync"
	"time"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
	"github.com/sirupsen/logrus"

	"github.com/dennwc/dom/net/ws"
	"github.com/salvacorts/eaopt"
	"google.golang.org/grpc"
)

// Log is the Pool Logger
var Log = logrus.New()

// PoolModel is a pool-based evolutionary algorithm
type PoolModel struct {
	Rnd            *rand.Rand
	KeepBest       bool
	ExtraOperators []eaopt.ExtraOperator
	EarlyStop      func(*PoolModel) bool

	// Sorting params
	SortFunction   func([]eaopt.Individual, float64)
	SortPrecission float64

	// Callbacks
	BestCallback       func(*PoolModel)
	GenerationCallback func(*PoolModel)

	CrossRate float64
	MutRate   float64

	PopSize        int
	MaxEvaluations int
	BestSolution   eaopt.Individual

	population *Population

	popSemaphore semaphore

	// Clients communications
	evaluations       int
	wg                sync.WaitGroup
	stop              chan bool
	shrinkChan        chan bool
	evaluationChannel chan eaopt.Individual
	evaluatedChannel  chan eaopt.Individual

	// Client Settings
	Delegate   ServiceDelegate
	grpcServer *grpc.Server
	grpcPort   int

	// Cluster communications
	cluster Cluster

	// Stats
	fitAvg float64
}

// MakePool Creates a new pool with default configuration
func MakePool(popSize int, grpcPort, clusterPort int, boostrapPeers []string, rnd *rand.Rand) *PoolModel {
	pool := &PoolModel{
		Rnd:               rnd,
		PopSize:           popSize,
		population:        MakePopulation(),
		popSemaphore:      make(semaphore, popSize),
		evaluations:       0,
		stop:              make(chan bool),
		evaluationChannel: make(chan eaopt.Individual, popSize),
		evaluatedChannel:  make(chan eaopt.Individual, popSize),
		grpcPort:          grpcPort,
		BestSolution: eaopt.Individual{
			Fitness: math.Inf(1),
		},

		cluster: Cluster{
			Logger:        Log,
			ListeningPort: clusterPort,
			BoostrapPeers: boostrapPeers,
		},
	}

	for i := 0; i < pool.PopSize; i++ {
		indi := eaopt.NewIndividual(mlp.NewRandMLP(pool.Rnd), pool.Rnd)
		pool.population.Add(indi)
	}

	return pool
}

// Minimize runs the algorithm by connecting the pipes
func (mod *PoolModel) Minimize() {
	// Wait for goroutins to end at the end of the function
	mod.wg.Add(4)
	defer mod.wg.Wait()

	// Launch client requests handlers
	go mod.handleEvaluate()
	go mod.handleEvaluated()

	// Launch cluster membership service
	go mod.cluster.Join(NodeMeta{
		GrpcPort:   int64(mod.grpcPort),
		GrpcWsPort: int64(mod.grpcPort + 1),
	})

	// Append all individuals in the population to the evaluation channel
	mod.population.Fold(func(indiv eaopt.Individual) bool {
		mod.evaluationChannel <- indiv

		return true
	})

	// Wait until all individuals have been evaluated
	mod.popSemaphore.Acquire(mod.PopSize)
	go mod.populationGrowControl()
	mod.popSemaphore.Release(mod.PopSize)

	for mod.evaluations < mod.MaxEvaluations {
		var offsprings []eaopt.Individual

		// take here randomly from population
		offsprings = mod.selection(4, 4)
		offsprings = mod.crossover(offsprings)
		offsprings = mod.mutate(offsprings)

		// Apply extra operators
		for i := range offsprings {
			for _, op := range mod.ExtraOperators {
				if mod.Rnd.Float64() <= op.Probability {
					offsprings[i].Evaluated = false
					offsprings[i].Genome = op.Operator(offsprings[i].Genome, mod.Rnd)
				}
			}
		}

		// Append non evaluated offspings to the channel and
		// Removing them from the population
		for i := range offsprings {
			if !offsprings[i].Evaluated {
				mod.population.Remove(offsprings[i])
				mod.evaluationChannel <- offsprings[i]
			} else {
				mod.popSemaphore.Release(1)
			}
		}
	}
}

// Binary torunament
func (mod *PoolModel) selection(nOffstrings, nCandidates int) []eaopt.Individual {
	offsprings := make([]eaopt.Individual, nOffstrings)

	// At least nOffsprings shold be available in the population
	// tu run this operator
	mod.popSemaphore.Acquire(nOffstrings)

	for i := range offsprings {
		alreadySelected := make(map[string]int) // 0 means not selected
		candidates := make([]eaopt.Individual, nCandidates)

		// Search n different candidates
		for j := range candidates {
			r, err := mod.population.RandIndividual(mod.Rnd)
			if err != nil {
				Log.Fatalf("Error on RandIndividual. %s\n", err.Error())
			}

			for alreadySelected[r.ID] == 1 {
				r, err = mod.population.RandIndividual(mod.Rnd)
				if err != nil {
					Log.Fatalf("Error on RandIndividual. %s\n", err.Error())
				}
			}

			alreadySelected[r.ID] = 1
			candidates[j] = r
		}

		mod.SortFunction(candidates, mod.SortPrecission)

		offsprings[i] = candidates[0].Clone(mod.Rnd)
		offsprings[i].ID = candidates[0].ID
	}

	return offsprings
}

// TODO: Get rid of odd arrays here
func (mod *PoolModel) crossover(in []eaopt.Individual) []eaopt.Individual {
	for i := 0; i < len(in)-1; i += 2 {
		if mod.Rnd.Float64() < mod.CrossRate {
			o1, o2 := in[i].Clone(mod.Rnd), in[i+1].Clone(mod.Rnd)

			o1.Genome.Crossover(o2.Genome, mod.Rnd)
			o1.Evaluated = false
			o2.Evaluated = false

			// Add keep best here? No, since you would have to evaluate the offsprings
			in[i] = o1
			in[i+1] = o2
		}
	}

	return in
}

func (mod *PoolModel) mutate(in []eaopt.Individual) []eaopt.Individual {
	for i := range in {
		if mod.Rnd.Float64() < mod.MutRate {
			in[i].Evaluated = false
			in[i].Genome.Mutate(mod.Rnd)
		}
	}

	return in
}

func (mod *PoolModel) handleEvaluated() {
	defer mod.wg.Done()

	for {
		select {
		default:
			o1, ok := <-mod.evaluatedChannel
			if !ok {
				Log.Debugln("in was already closed")
				return
			}

			// Update best individual
			if o1.Fitness < mod.BestSolution.Fitness {
				mod.BestSolution = o1.Clone(mod.Rnd)
				mod.BestSolution.ID = o1.ID

				if mod.BestCallback != nil {
					mod.BestCallback(mod)
				}
			}

			mod.population.Add(o1)
			mod.popSemaphore.Release(1)
			mod.evaluations++

			if mod.evaluations%100 == 0 {
				fmt.Printf("Best Fitness: %f\n", mod.BestSolution.Fitness)
				fmt.Printf("Avg Fitness: %f\n", mod.GetAverageFitness())
				fmt.Printf("[%d / %d] Population [%d]: ", mod.evaluations, mod.MaxEvaluations, mod.population.Length())
				mod.population.Fold(func(i eaopt.Individual) bool {
					fmt.Printf("%.2f, ", i.Fitness)
					return true
				})
				fmt.Printf("\n\n")
			}

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

func (mod *PoolModel) handleEvaluate() {
	defer mod.wg.Done()

	var (
		listenAddr = "127.0.0.1"
		nativePort = mod.grpcPort
		wasmPort   = nativePort + 1
	)

	mlpServer := &Server{
		Input:  mod.evaluationChannel,
		Output: mod.evaluatedChannel,
		Log:    Log,

		// For stats
		Pool: mod,
	}

	// Start listening on server
	mod.grpcServer = grpc.NewServer()
	RegisterDistributedEAServer(mod.grpcServer, mlpServer)

	// Setup listener for native clients
	lisNative, err := net.Listen("tcp", fmt.Sprintf("%s:%d", listenAddr, nativePort))
	if err != nil {
		Log.Fatalf("Failed to listen: %v", err)
	}

	// Setup listener for wasm clients based on websockets
	lisWasm, err := ws.Listen(fmt.Sprintf("ws://%s:%d", listenAddr, wasmPort), nil)
	if err != nil {
		panic(err)
	}
	defer lisNative.Close()

	Log.Infof("gRPC Listening on %s", lisNative.Addr().String())
	go mod.grpcServer.Serve(lisNative)

	Log.Infof("gRPC-Web Listening on %s:%d", lisWasm.Addr().String(), wasmPort)
	go mod.grpcServer.Serve(lisWasm)

	// Wait until stop is received
	select {
	case <-mod.stop:
	}
}

func getFunctionName(i interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
}

// GetTotalEvaluations returns the total number of evaluations carried out so far
func (mod *PoolModel) GetTotalEvaluations() int {
	return mod.evaluations
}

// GetAverageFitness returns the vaerage fitness of the population
func (mod *PoolModel) GetAverageFitness() float64 {
	totalFitness := 0.0
	items := 0

	mod.population.Fold(func(indiv eaopt.Individual) bool {
		totalFitness += indiv.Fitness
		items++
		return true
	})

	return totalFitness / float64(items)
}

// Reduce the population size to a fixed preset number by elimiating the
// individuals with lower fitness
func (mod *PoolModel) populationGrowControl() {
	defer mod.wg.Done()

	for {
		time.Sleep(1 * time.Second)

		snap := mod.population.Snapshot()
		indivArr := make([]eaopt.Individual, len(snap))
		var bestCopy eaopt.Individual
		i := 0

		for _, item := range snap {
			indivArr[i] = item.Object.(eaopt.Individual)
			i++
		}

		mod.SortFunction(indivArr, mod.SortPrecission)

		for i := mod.PopSize; i < len(indivArr); i++ {
			mod.population.Remove(indivArr[i])
		}

		// Replace the remaining worst one by the best one so far
		if len(indivArr) >= mod.PopSize {
			mod.population.Remove(indivArr[mod.PopSize-1])

			bestCopy = mod.BestSolution.Clone(mod.Rnd)
			mod.population.Add(bestCopy)
		}
	}
}
