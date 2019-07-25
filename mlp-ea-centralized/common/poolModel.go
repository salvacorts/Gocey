package common

import (
	"math"
	"math/rand"
	"sync"

	"github.com/salvacorts/eaopt"
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

// Init initialized the model by creating a pipeline of channels with the different
// genetic operators
func (mod *PipedPoolModel) init() {
	// Append Evaluate, Cross and Mutate
	mod.pipeline = append(mod.pipeline, mod.evaluate, mod.crossover, mod.mutate)

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

	for i := 0; i < len(mod.pipeline)-1; i++ {
		current := make(chan eaopt.Individual, mod.PopSize)

		go mod.pipeline[i](previous, current)

		previous = current
	}

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

				o1.ID, o2.ID = p1.ID, p2.ID

				if mod.KeepBest {
					indis := []eaopt.Individual{p1, p2, o1, o2}

					mod.SortFunction(indis)

					out <- indis[0]
					out <- indis[1]
				} else {
					out <- o1
					out <- o2
				}
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

func (mod *PipedPoolModel) evaluate(in, out chan eaopt.Individual) {
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

			if !o1.Evaluated {
				o1.Evaluate()

				// Calculate average fitness here
				mod.fitAvg += o1.Fitness
				mod.fitAvg /= 2

				if o1.Fitness < mod.BestSolution.Fitness {
					mod.BestSolution = o1.Clone(mod.Rnd)

					if mod.BestCallback != nil {
						mod.BestCallback(mod)
					}
				}
			}

			info := mod.population[o1.ID]
			info.Generation++

			if info.Generation > mod.Generation+1 {
				mod.Generation++

				if mod.GenerationCallback != nil {
					mod.GenerationCallback(mod)
				}

				if mod.Generation >= mod.MaxGeneration || mod.EarlyStop(mod) {
					select {
					case <-mod.stop:
						Log.Debugln("mod.stop was already closed")
					default:
						if mod.stop != nil {
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

// FitAvg returns the average fitness of the population
func (mod *PipedPoolModel) FitAvg() float64 {
	return mod.fitAvg
}
