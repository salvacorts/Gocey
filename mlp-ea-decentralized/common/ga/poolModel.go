//+build !js

package ga

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"net"
	"reflect"
	"runtime"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/dennwc/dom/net/ws"
	"github.com/salvacorts/eaopt"
	"google.golang.org/grpc"
)

// Log is the Pool Logger
var Log = logrus.New()

// SortFunction gets an array of individuals and sorts them with a given precission
type SortFunction = func(slice []eaopt.Individual, precission int) []eaopt.Individual

// GenomeGenerator is a function that creates genomes randomly initializaed
type GenomeGenerator = func(*rand.Rand) eaopt.Genome

// PoolModel is a pool-based evolutionary algorithm
// 		TODO: Wrap configuration into ConfigTypes (e.g. ClientConfig, IslandConfig, ModelConfig...)
type PoolModel struct {
	Rnd            *rand.Rand
	KeepBest       bool
	ExtraOperators []eaopt.ExtraOperator
	EarlyStop      func(*PoolModel) bool

	// Sorting params
	SortFunc       SortFunction
	SortPrecission int

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

	// Server communications
	BroadcastedIndividual chan Individual
	cluster               *Cluster
	NMigrate              int

	// Stats
	metricsServer *MetricsServer
}

// MakePool Creates a new pool with default configuration
func MakePool(
	popSize int, grpcPort, clusterPort, metricsPort int, boostrapPeers []string,
	rnd *rand.Rand, generator GenomeGenerator) *PoolModel {
	pool := &PoolModel{
		Rnd:                   rnd,
		PopSize:               popSize,
		population:            MakePopulation(),
		popSemaphore:          make(semaphore, popSize),
		evaluations:           0,
		stop:                  make(chan bool),
		evaluationChannel:     make(chan eaopt.Individual, popSize),
		evaluatedChannel:      make(chan eaopt.Individual, popSize),
		BroadcastedIndividual: make(chan Individual, popSize),
		grpcPort:              grpcPort,
		BestSolution: eaopt.Individual{
			Fitness: math.Inf(1),
		},
	}

	pool.metricsServer = MakeMetricsServer(pool, fmt.Sprintf("0.0.0.0:%d", metricsPort), logrus.New())

	pool.cluster = MakeCluster(clusterPort, popSize, pool.BroadcastedIndividual, boostrapPeers, logrus.New())
	pool.cluster.Logger.SetLevel(logrus.DebugLevel)

	for i := 0; i < pool.PopSize; i++ {
		indi := eaopt.NewIndividual(generator(pool.Rnd), pool.Rnd)
		pool.population.Add(indi)
	}

	return pool
}

// Minimize runs the algorithm by connecting the pipes
func (pool *PoolModel) Minimize() {
	// Wait for goroutins to end at the end of the function
	defer pool.wg.Wait()
	defer close(pool.shrinkChan)
	defer close(pool.evaluationChannel)
	defer close(pool.evaluatedChannel)
	defer close(pool.BroadcastedIndividual)
	defer close(pool.stop)

	// Launch client requests handlers
	go pool.handleEvaluate()
	go pool.handleEvaluated()
	go pool.handleBroadcastedIndividual()

	// Launch cluster membership service
	go pool.cluster.Start(pool.getCurrentNodeMetadata())
	defer pool.cluster.Shutdown()

	// Launch metrics server
	go pool.metricsServer.Start()

	// Append all individuals in the population to the evaluation channel
	pool.population.Fold(func(indiv eaopt.Individual) bool {
		pool.evaluationChannel <- indiv
		return true
	})

	// Wait until all individuals have been evaluated
	pool.popSemaphore.Acquire(pool.PopSize)
	go pool.populationGrowControl()
	go pool.migrationScheduler()
	pool.popSemaphore.Release(pool.PopSize)

	for pool.evaluations < pool.MaxEvaluations {
		var offsprings []eaopt.Individual

		// At least nOffsprings shold be available in the population
		// tu run this operator
		pool.popSemaphore.Acquire(4)

		// take here randomly from population
		offsprings = pool.selection(4, 4)
		offsprings = pool.crossover(offsprings)
		offsprings = pool.mutate(offsprings)

		// Apply extra operators
		for i := range offsprings {
			for _, op := range pool.ExtraOperators {
				if pool.Rnd.Float64() <= op.Probability {
					offsprings[i].Genome = op.Operator(offsprings[i].Genome, pool.Rnd)
					offsprings[i].Evaluated = false
				}
			}
		}

		// Append non evaluated offspings to the channel and
		// Removing them from the population
		for i := range offsprings {
			if !offsprings[i].Evaluated {
				pool.population.Remove(offsprings[i])
				pool.evaluationChannel <- offsprings[i]
			} else {
				pool.popSemaphore.Release(1)
			}
		}
	}
}

// Selection selects nOffstrings on a tournamnet of nCandidates
func (pool *PoolModel) selection(nOffstrings, nCandidates int) []eaopt.Individual {
	offsprings := make([]eaopt.Individual, nOffstrings)

	for i := range offsprings {
		alreadySelected := make(map[string]int) // 0 means not selected
		candidates := make([]eaopt.Individual, nCandidates)

		// Search n different candidates
		for j := range candidates {
			r, err := pool.population.RandIndividual(pool.Rnd)
			if err != nil {
				Log.Fatalf("Error on RandIndividual. %s\n", err.Error())
			}

			for alreadySelected[r.ID] == 1 {
				r, err = pool.population.RandIndividual(pool.Rnd)
				if err != nil {
					Log.Fatalf("Error on RandIndividual. %s\n", err.Error())
				}
			}

			alreadySelected[r.ID] = 1
			candidates[j] = r
		}

		candidates = pool.SortFunc(candidates, pool.SortPrecission)

		offsprings[i] = candidates[0].Clone(pool.Rnd)
		offsprings[i].ID = candidates[0].ID
	}

	return offsprings
}

func (pool *PoolModel) crossover(in []eaopt.Individual) []eaopt.Individual {
	offsprings := make([]eaopt.Individual, len(in))
	for i := 0; i < len(in)-1; i += 2 {
		if pool.Rnd.Float64() < pool.CrossRate {
			o1, o2 := in[i].Clone(pool.Rnd), in[i+1].Clone(pool.Rnd)

			o1.Genome.Crossover(o2.Genome, pool.Rnd)
			o1.Evaluated = false
			o2.Evaluated = false

			// Add keep best here? No, since you would have to evaluate the offsprings
			offsprings[i] = o1
			offsprings[i+1] = o2
		} else {
			offsprings[i] = in[i]
			offsprings[i+1] = in[i+1]
		}
	}

	return offsprings
}

func (pool *PoolModel) mutate(in []eaopt.Individual) []eaopt.Individual {
	for i := range in {
		if pool.Rnd.Float64() < pool.MutRate {
			in[i].Genome.Mutate(pool.Rnd)
			in[i].Evaluated = false
		}
	}

	return in
}

func (pool *PoolModel) handleEvaluated() {
	pool.wg.Add(1)
	defer pool.wg.Done()

	Log.Infoln("handleEvaluated goroutine started")

	for {
		select {
		default:
			o1, ok := <-pool.evaluatedChannel
			if !ok {
				Log.Debugln("in was already closed")
				return
			}

			// Update best individual
			if o1.Fitness < pool.BestSolution.Fitness {
				pool.BestSolution = o1.Clone(pool.Rnd)
				pool.BestSolution.ID = o1.ID

				pool.metricsServer.BestFitnessGauge.Set(
					pool.BestSolution.Fitness)

				// Spread the new best solution accross the cluster
				indiv := Individual{
					IndividualID: pool.BestSolution.ID,
					Evaluated:    pool.BestSolution.Evaluated,
					Fitness:      pool.BestSolution.Fitness,
					Genome: pool.Delegate.SerializeGenome(
						pool.BestSolution.Genome),
				}
				pool.cluster.BroadcastBestIndividual <- indiv
				pool.metricsServer.OutgoingBroadcasts.Inc()

				if pool.BestCallback != nil {
					pool.BestCallback(pool)
				}
			}

			pool.population.Add(o1)
			pool.popSemaphore.Release(1)
			pool.evaluations++
			pool.metricsServer.EvaluationsCount.Inc()

			if pool.evaluations%100 == 0 {
				fmt.Printf("Best Fitness: %f\n", pool.BestSolution.Fitness)
				fmt.Printf("Avg Fitness: %f\n", pool.GetAverageFitness())
				fmt.Printf("[%d / %d] Population [%d]: ", pool.evaluations, pool.MaxEvaluations, pool.population.Length())
				pool.population.Fold(func(i eaopt.Individual) bool {
					fmt.Printf("%.2f, ", i.Fitness)
					return true
				})
				fmt.Printf("\n\n")
			}

			if pool.evaluations >= pool.MaxEvaluations ||
				(pool.EarlyStop != nil && pool.EarlyStop(pool)) {
				select {
				case <-pool.stop:
					Log.Debugln("pool.stop was already closed")
				default:
					if pool.stop != nil {
						pool.grpcServer.Stop()
						close(pool.stop)
					}
				}
			}

		case <-pool.stop:
			return
		}
	}
}

func (pool *PoolModel) handleEvaluate() {
	pool.wg.Add(1)
	defer pool.wg.Done()

	Log.Infoln("handleEvaluate goroutine started")

	var (
		listenAddr = "0.0.0.0"
		nativePort = pool.grpcPort
		wasmPort   = nativePort + 1
	)

	mlpServer := &Server{
		Input:  pool.evaluationChannel,
		Output: pool.evaluatedChannel,
		Log:    Log,

		// For stats
		Pool: pool,
	}

	// Start listening on server
	pool.grpcServer = grpc.NewServer()
	RegisterDistributedEAServer(pool.grpcServer, mlpServer)
	defer pool.grpcServer.Stop()

	// Setup listener for native clients
	lisNative, err := net.Listen("tcp", fmt.Sprintf("%s:%d", listenAddr, nativePort))
	if err != nil {
		Log.Fatalf("Failed to listen: %v", err)
	}
	defer lisNative.Close()

	// Setup listener for wasm clients based on websockets
	lisWasm, err := ws.Listen(fmt.Sprintf("ws://%s:%d", listenAddr, wasmPort), nil)
	if err != nil {
		panic(err)
	}
	defer lisWasm.Close()

	Log.Infof("gRPC Listening on %s", lisNative.Addr().String())
	go pool.grpcServer.Serve(lisNative)

	Log.Infof("gRPC-Web Listening on %s:%d", lisWasm.Addr().String(), wasmPort)
	go pool.grpcServer.Serve(lisWasm)

	// Wait until stop is received
	select {
	case <-pool.stop:
		return
	}
}

func (pool *PoolModel) handleBroadcastedIndividual() {
	pool.wg.Add(1)
	defer pool.wg.Done()

	Log.Infoln("handleBroadcastedIndividual goroutine started")

	for {
		select {
		default:
			in, ok := <-pool.BroadcastedIndividual
			if !ok {
				Log.Debugln("pool.BroadcastedIndividual was already closed")
				return
			}

			pool.metricsServer.IncomingBroadcasts.Inc()

			indiv := eaopt.Individual{
				ID:        in.IndividualID,
				Fitness:   in.Fitness,
				Evaluated: in.Evaluated,
				Genome:    pool.Delegate.DeserializeGenome(in.Genome),
			}

			if indiv.Fitness < pool.BestSolution.Fitness {
				Log.Infof("Got new best solution from gossip. Now: %.2f", indiv.Fitness)
				pool.BestSolution = indiv
			} else {
				Log.Warn("Got broadcasted solution worse than currest best one. Ignoring...")
			}

		case <-pool.stop:
			return
		}
	}
}

func getFunctionName(i interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
}

func (pool *PoolModel) getCurrentNodeMetadata() NodeMetadata {
	return NodeMetadata{
		GrpcPort:   int64(pool.grpcPort),
		GrpcWsPort: int64(pool.grpcPort + 1),
	}
}

// GetTotalEvaluations returns the total number of evaluations carried out so far
func (pool *PoolModel) GetTotalEvaluations() int {
	return pool.evaluations
}

// GetPopulationSnapshot returns an snapshot of the population at a given moment
func (pool *PoolModel) GetPopulationSnapshot() []eaopt.Individual {
	snapMap := pool.population.Snapshot()
	snap := make([]eaopt.Individual, 0, len(snapMap))

	for _, v := range snapMap {
		snap = append(snap, v.Object.(eaopt.Individual))
	}

	return snap
}

// GetAverageFitness returns the vaerage fitness of the population
func (pool *PoolModel) GetAverageFitness() float64 {
	totalFitness := 0.0
	items := 0

	pool.population.Fold(func(indiv eaopt.Individual) bool {
		totalFitness += indiv.Fitness
		items++
		return true
	})

	return totalFitness / float64(items)
}

// Reduce the population size to a fixed preset number by elimiating the
// individuals with lower fitness
func (pool *PoolModel) populationGrowControl() {
	pool.wg.Add(1)
	defer pool.wg.Done()

	Log.Infoln("populationGrowControl goroutine started")

	for {
		select {
		default:
			time.Sleep(1 * time.Second)

			snap := pool.population.Snapshot()
			indivArr := make([]eaopt.Individual, len(snap))
			var bestCopy eaopt.Individual
			i := 0

			for _, item := range snap {
				indivArr[i] = item.Object.(eaopt.Individual)
				i++
			}

			indivArr = pool.SortFunc(indivArr, pool.SortPrecission)

			for i := pool.PopSize; i < len(indivArr); i++ {
				pool.population.Remove(indivArr[i])
			}

			// Replace the remaining worst one by the best one so far
			if len(indivArr) >= pool.PopSize {
				pool.population.Remove(indivArr[pool.PopSize-1])

				bestCopy = pool.BestSolution.Clone(pool.Rnd)
				pool.population.Add(bestCopy)
			}

		case <-pool.stop:
			return
		}

	}
}

// Migrate individuals to another population at a given interval
func (pool *PoolModel) migrationScheduler() {
	pool.wg.Add(1)
	defer pool.wg.Done()

	Log.Infoln("migrationScheduler goroutine started")

	for {
		select {
		default:
			time.Sleep(5 * time.Second)
			var conn *grpc.ClientConn
			dialed := false

			indivArr := pool.selection(pool.NMigrate, 4)
			migrate := make([]Individual, pool.NMigrate)
			for i := range migrate {
				migrate[i] = Individual{
					IndividualID: indivArr[i].ID,
					Fitness:      indivArr[i].Fitness,
					Evaluated:    indivArr[i].Evaluated,
					Genome:       pool.Delegate.SerializeGenome(indivArr[i].Genome),
				}
			}

			// Pick a random node from the cluster and dial it
			members := pool.cluster.GetMembers()
			for !dialed {
				remotePeer := members[pool.Rnd.Intn(len(members))]
				if remotePeer == pool.cluster.list.LocalNode() {

					// If it is only me in the cluster, break and wait
					if pool.cluster.GetNumNodes() == 1 {
						break
					}

					continue
				}

				remoteMetadata := NodeMetadata{}

				err := remoteMetadata.Unmarshal(remotePeer.Meta)
				if err != nil {
					Log.Errorf("Could not parse Node Metadata on migrationScheduler. %s", err.Error())
				}

				remoteAddr := fmt.Sprintf("%s:%d",
					remotePeer.Addr.String(), remoteMetadata.GrpcPort)

				conn, err = grpc.Dial(remoteAddr, grpc.WithInsecure())
				if err != nil {
					Log.Warnf("Could not dial %s. %s", remoteAddr, err.Error())
				}

				dialed = true
			}

			if !dialed {
				continue
			}

			client := NewDistributedEAClient(conn)
			if _, err := client.MigrateIndividuals(context.Background(), &IndividualsBatch{
				Individuals: migrate,
			}); err != nil {
				Log.Errorf("could not MigrateIndividuals to %s. %s", conn.Target(), err.Error())
			}

			pool.metricsServer.OutgoingMigrationsCount.Inc()

			conn.Close()

		case <-pool.stop:
			return
		}

	}
}
