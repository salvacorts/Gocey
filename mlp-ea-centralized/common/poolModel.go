package common

import (
	"math"
	"math/rand"

	"github.com/salvacorts/eaopt"
)

// Pipe represents a function that applies an operator to a Genome that
// comes from the first channel and goes through the second channel
type Pipe = func(chan eaopt.Individual, chan eaopt.Individual)

// PipedPoolModel is a evolutionary algorithm model that uses channels to apply
// genetic operators over a population
type PipedPoolModel struct {
	Rnd            *rand.Rand
	KeepBest       bool
	SortFunction   func(eaopt.Individuals)
	ExtraOperators []eaopt.ExtraOperator
	Callback       func(PipedPoolModel)

	CrossRate float64
	MutRate   float64

	PopSize      uint
	Generations  uint
	BestSolution eaopt.Individual
	Population   eaopt.Individuals

	pipeline []Pipe
}

// Init initialized the model by creating a pipeline of channels with the different
// genetic operators
func (mod *PipedPoolModel) init() {
	// Append Evaluate, Cross and Mutate
	mod.pipeline = append(mod.pipeline, mod.evaluate, mod.crossover, mod.mutate)

	// Append the rest of extra operators
	for _, operator := range mod.ExtraOperators {
		mod.pipeline = append(mod.pipeline, func(in, out chan eaopt.Individual) {
			o1 := <-in

			if mod.Rnd.Float64() < operator.Probability {
				o1.ApplyExtraOperator(operator, mod.Rnd)
			}

			out <- o1
		})
	}

	// Create initial population
	mod.Population = make(eaopt.Individuals, mod.PopSize)

	for i := range mod.Population {
		mod.Population[i] = eaopt.NewIndividual(
			NewRandMLP(mod.Rnd), mod.Rnd)
	}

	mod.BestSolution.Fitness = math.Inf(1)
}

// Minimize runs the algorithm by connecting the pipes
func (mod PipedPoolModel) Minimize() {
	mod.init()

	// Connect the pipeline creating a circular pipeline
	start := make(chan eaopt.Individual)
	previous := start

	for i := 0; i < len(mod.pipeline)-1; i++ {
		current := make(chan eaopt.Individual)

		go mod.pipeline[i](previous, current)

		previous = current
	}

	// Connect the last pipe to the first one
	go mod.pipeline[len(mod.pipeline)-1](previous, start)

	// Put all individuals of the population in the first channel
	for i := range mod.Population {
		start <- mod.Population[i]
	}

}

func (mod PipedPoolModel) crossover(in, out chan eaopt.Individual) {
	// Get parents from the input channel
	p1, p2 := <-in, <-in

	if mod.Rnd.Float64() < mod.CrossRate {

		// Create offsprings
		o1, o2 := p1.Clone(mod.Rnd), p2.Clone(mod.Rnd)
		o1.Crossover(o2, mod.Rnd)

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
}

func (mod PipedPoolModel) mutate(in, out chan eaopt.Individual) {
	o1 := <-in

	if mod.Rnd.Float64() < mod.MutRate {
		o1.Mutate(mod.Rnd)
	}

	out <- o1
}

func (mod PipedPoolModel) evaluate(in, out chan eaopt.Individual) {
	o1 := <-in

	o1.Evaluate()

	if o1.Fitness < mod.BestSolution.Fitness {
		mod.BestSolution = o1

		mod.Callback(mod)
	}

	out <- o1
}
