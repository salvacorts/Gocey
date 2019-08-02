package common

import (
	"fmt"
	"math"
	"math/rand"
	"net"
	"reflect"
	"runtime"
	"sync"
	"time"

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
	WorstSolution  eaopt.Individual

	//population    sync.Map
	population    map[string]eaopt.Individual
	currentPopLen int
	pipeline      []Pipe

	popSemaphore semaphore
	//popCountMutex   *sync.Mutex
	populationMutex *sync.Mutex

	evaluations       int
	wg                sync.WaitGroup
	grpcServer        *grpc.Server
	stop              chan bool
	shrinkChan        chan bool
	evaluationChannel chan eaopt.Individual
	evaluatedChannel  chan eaopt.Individual

	// Stats
	fitAvg float64
}

// MakePool Creates a new pool with default configuration
func MakePool(popSize int, rnd *rand.Rand) *PipedPoolModel {
	pool := &PipedPoolModel{
		Rnd:          rnd,
		PopSize:      popSize,
		population:   make(map[string]eaopt.Individual),
		popSemaphore: make(semaphore, popSize),
		//popCountMutex:     &sync.Mutex{},
		populationMutex:   &sync.Mutex{},
		evaluations:       0,
		stop:              make(chan bool),
		evaluationChannel: make(chan eaopt.Individual, popSize),
		evaluatedChannel:  make(chan eaopt.Individual, popSize),
		BestSolution: eaopt.Individual{
			Fitness: math.Inf(1),
		},
		WorstSolution: eaopt.Individual{
			Fitness: math.Inf(-1),
		},
	}

	for i := 0; i < pool.PopSize; i++ {
		indi := eaopt.NewIndividual(NewRandMLP(pool.Rnd), pool.Rnd)
		//pool.population.Store(indi.ID, indi)
		pool.population[indi.ID] = indi
	}

	// pool.popCountMutex.Lock()
	// pool.currentPopLen = pool.PopSize
	// pool.popCountMutex.Unlock()

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

	// fmt.Printf("\n\nPopulation: ")
	// for i := range mod.population {
	// 	fmt.Printf("%s, ", mod.population[i].ID)
	// }
	// fmt.Printf("\n")

	go mod.handleEvaluate()
	go mod.handleEvaluated()

	for k := range mod.population {
		mod.evaluationChannel <- mod.population[k]
	}

	// Wait until all individuals have been evaluated
	mod.popSemaphore.Acquire(mod.PopSize)
	mod.shrinkChan = make(chan bool)
	go mod.populationGrowControl()
	mod.popSemaphore.Release(mod.PopSize)

	for mod.evaluations < mod.MaxEvaluations {
		// take here randomly from population
		offsprings = mod.selection(4)
		offsprings = mod.crossover(offsprings)
		offsprings = mod.mutate(offsprings)

		// Apply extra operators
		for i := range offsprings {
			for _, op := range mod.ExtraOperators {
				if mod.Rnd.Float64() <= op.Probability {
					//mod.removeIndividual(offsprings[i])

					offsprings[i].Evaluated = false
					offsprings[i].Genome = op.Operator(offsprings[i].Genome, mod.Rnd)
				}
			}
		}

		// Append non evaluated offspings to the channel and
		// evaluated one again to the population
		for i := range offsprings {
			if !offsprings[i].Evaluated {
				mod.removeIndividual(offsprings[i])
				mod.evaluationChannel <- offsprings[i]
			} else {
				//mod.population.Store(offsprings[i].ID, offsprings[i])
				//mod.addIndividual(offsprings[i])
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
	// if len(mod.population) < n {
	// 	Log.Fatal("Not enought eaopt.Individuals on the population to select")
	// }

	for i := range offsprings {
		rnd1 := mod.getRandOnMap()
		rnd2 := mod.getRandOnMap()

		// Make them differrent.
		for rnd1.ID == rnd2.ID {
			rnd2 = mod.getRandOnMap()
		}

		if rnd1.Fitness < rnd2.Fitness {
			offsprings[i] = rnd1.Clone(mod.Rnd)
			offsprings[i].ID = rnd1.ID // Maintain ID from clone to be able to replace it if not deleted
		} else {
			offsprings[i] = rnd2.Clone(mod.Rnd)
			offsprings[i].ID = rnd2.ID
		}
	}

	return offsprings
}

// TODO: Get rid of odd arrays here
func (mod *PipedPoolModel) crossover(in []eaopt.Individual) []eaopt.Individual {
	//new := make([]eaopt.Individual, len(in))

	// Copy all the offsprings
	// for i := range new {
	// 	new[i] = in[i]
	// }

	// Sort clones
	// SortByFitnessAndNeurons(new)

	// With the offspring first half, replace the second half
	// for i := 0; i < len(new)/2; i += 2 {
	// 	o1, o2 := new[i].Clone(mod.Rnd), new[i+1].Clone(mod.Rnd)
	// 	o1.Genome.Crossover(o2.Genome, mod.Rnd)
	// 	o1.Evaluated = false
	// 	o2.Evaluated = false

	// 	new[len(new)-1-i] = o1
	// 	new[len(new)-1-(i+1)] = o2
	// }

	for i := 0; i < len(in)-1; i += 2 {
		if mod.Rnd.Float64() < mod.CrossRate {
			// Delete the modified individuals from the population
			// since they are no longer evaluated
			// mod.removeIndividual(in[i])
			// mod.removeIndividual(in[i+1])
			// delete(mod.population, in[i].ID)
			// delete(mod.population, in[i+1].ID)

			o1, o2 := in[i].Clone(mod.Rnd), in[i+1].Clone(mod.Rnd)

			o1.Genome.Crossover(o2.Genome, mod.Rnd)
			o1.Evaluated = false
			o2.Evaluated = false

			// TODO: Add keep best here
			in[i] = o1
			in[i+1] = o2
		}
	}

	return in
}

func (mod *PipedPoolModel) mutate(in []eaopt.Individual) []eaopt.Individual {
	for i := range in {
		if mod.Rnd.Float64() < mod.MutRate {
			// mod.removeIndividual(in[i])
			//  []eaopt.Individual(mod.population, in[i].ID)

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

			// Update best individual
			if o1.Fitness < mod.BestSolution.Fitness {
				mod.BestSolution = o1.Clone(mod.Rnd)
				mod.BestSolution.ID = o1.ID

				mod.populationMutex.Lock()
				if mod.BestCallback != nil {
					mod.BestCallback(mod)
				}

				fmt.Printf("\n\nPopulation [%d]: ", len(mod.population))
				for i := range mod.population {
					fmt.Printf("%s, ", mod.population[i].ID)
				}
				fmt.Printf("\n")
				mod.populationMutex.Unlock()
			}

			// mod.population.Store(o1.ID, o1)
			// mod.popCountMutex.Lock()
			// mod.currentPopLen++
			// mod.popCountMutex.Unlock()

			mod.addIndividual(o1)
			mod.popSemaphore.Release(1)
			mod.evaluations++

			// Garbage collect worst individuals from the population
			// if mod.evaluations%25 == 0 {
			// 	if mod.shrinkChan != nil {
			// 		mod.shrinkChan <- true
			// 	}
			// }

			if mod.evaluations%250 == 0 {
				fmt.Printf("Best Fitness: %f\n", mod.BestSolution.Fitness)
				fmt.Printf("Avg Fitness: %f\n", mod.GetAverageFitness())
				fmt.Printf("[%d / %d] Population [%d]: ", mod.evaluations, mod.MaxEvaluations, len(mod.population))
				mod.populationMutex.Lock()
				for i := range mod.population {
					fmt.Printf("%s, ", mod.population[i].ID)
				}
				fmt.Printf("\n\n")
				mod.populationMutex.Unlock()
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

func (mod *PipedPoolModel) removeIndividual(indiv eaopt.Individual) {
	mod.populationMutex.Lock()
	defer mod.populationMutex.Unlock()

	delete(mod.population, indiv.ID)
}

func (mod *PipedPoolModel) addIndividual(indiv eaopt.Individual) {
	mod.populationMutex.Lock()
	defer mod.populationMutex.Unlock()

	mod.population[indiv.ID] = indiv
}

func (mod *PipedPoolModel) getRandOnMap() (out eaopt.Individual) {
	mod.populationMutex.Lock()
	defer mod.populationMutex.Unlock()

	// mod.popCountMutex.Lock()
	// if mod.currentPopLen == 0 {
	// 	Log.Fatal("Empty population")
	// }
	// // rnd := mod.Rnd.Intn(mod.currentPopLen)
	// // idx := 0
	// mod.popCountMutex.Unlock()

	// // Range on the map until idx is reached. false makes the iteration stop
	// f := func(key, value interface{}) bool {
	// 	if idx == rnd {
	// 		out = value.(eaopt.Individual)
	// 		return false
	// 	}

	// 	idx++
	// 	return true
	// }

	// mod.population.Range(f)

	// if out.ID == "" {
	// 	mod.printPopulation()
	// 	fmt.Printf("Pop Len: %d", mod.currentPopLen)
	// 	Log.Fatalf("Could not get a random indiv from the population\n")
	// }

	rnd := mod.Rnd.Intn(len(mod.population))
	idx := 0

	for _, indiv := range mod.population {
		if idx == rnd {
			out = indiv
			break
		}

		idx++
	}

	return
}

func (mod *PipedPoolModel) printPopulation() {
	// len := 0

	// // Range on the map until idx is reached. false makes the iteration stop
	// f := func(key, value interface{}) bool {
	// 	fmt.Printf("%s, ", value.(eaopt.Individual).ID)
	// 	len++
	// 	return true
	// }

	// mod.population.Range(f)
	// fmt.Printf("\nLen: %d\n\n", len)
}

func (mod *PipedPoolModel) GetAverageFitness() float64 {
	mod.populationMutex.Lock()
	defer mod.populationMutex.Unlock()

	totalFitness := 0.0

	for _, indiv := range mod.population {
		totalFitness += indiv.Fitness
	}

	return totalFitness / float64(len(mod.population))

	// len := 0

	// // Range on the map until idx is reached. false makes the iteration stop
	// f := func(key, value interface{}) bool {
	// 	fmt.Printf("%s, ", value.(eaopt.Individual).ID)
	// 	len++
	// 	return true
	// }

	// mod.population.Range(f)
	// fmt.Printf("\nLen: %d\n\n", len)
}

// Reduce the population size to a fixed preset number by elimiating the
// individuals with lower fitness
func (mod *PipedPoolModel) populationGrowControl() {
	defer mod.wg.Done()

	for {
		time.Sleep(1)

		mod.populationMutex.Lock()
		indivArr := make([]eaopt.Individual, len(mod.population))
		i := 0

		// Get array from Map
		for _, indiv := range mod.population {
			indivArr[i] = indiv
			i++
		}

		SortByFitnessAndNeurons(indivArr)

		// Remove worst elements remaining
		for i := popSize; i < len(indivArr); i++ {
			delete(mod.population, indivArr[i].ID)
		}

		mod.populationMutex.Unlock()
	}
}
