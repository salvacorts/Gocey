package common

import (
	"math"
	"sort"

	"github.com/salvacorts/eaopt"
)

// SortByFitnessAndNeurons First sorts by Fitness and then by number of neurons
func SortByFitnessAndNeurons(indis eaopt.Individuals) {

	// Round with precission
	round := func(i float64) float64 {
		return math.Floor(i*Precission) / Precission
	}

	// Sort individuals first by fitness and then by number of neurons
	sort.Slice(indis, func(i, j int) bool {

		if round(indis[i].Fitness) == round(indis[j].Fitness) {
			neuronsI := 0
			neuronsJ := 0

			for _, l := range indis[i].Genome.(*MLP).NeuralLayers {
				neuronsI += l.Length
			}

			for _, l := range indis[j].Genome.(*MLP).NeuralLayers {
				neuronsJ += l.Length
			}

			return neuronsI < neuronsJ
		}

		return indis[i].Fitness < indis[j].Fitness
	})
}

func getSteadyStateModel() eaopt.Model {
	return eaopt.ModSteadyState{
		Selector:  eaopt.SelElitism{},
		KeepBest:  true,
		MutRate:   mutProb,
		CrossRate: crossProb,

		ExtraOperators: []eaopt.ExtraOperator{
			eaopt.ExtraOperator{Operator: AddNeuron, Probability: addNeuronProb},
			eaopt.ExtraOperator{Operator: RemoveNeuron, Probability: removeNeuronProb},
			eaopt.ExtraOperator{Operator: SubstituteNeuron, Probability: substituteNeuronProb},
			eaopt.ExtraOperator{Operator: Train, Probability: trainProb},
		},

		SortFunc: SortByFitnessAndNeurons,
	}
}

func getGenerationalModelElistism() eaopt.Model {
	return eaopt.ModGenerational{
		Selector:  eaopt.SelElitism{SortFunc: SortByFitnessAndNeurons},
		MutRate:   mutProb,
		CrossRate: crossProb,

		ExtraOperators: []eaopt.ExtraOperator{
			eaopt.ExtraOperator{Operator: AddNeuron, Probability: addNeuronProb},
			eaopt.ExtraOperator{Operator: RemoveNeuron, Probability: removeNeuronProb},
			eaopt.ExtraOperator{Operator: SubstituteNeuron, Probability: substituteNeuronProb},
			eaopt.ExtraOperator{Operator: Train, Probability: trainProb},
		},

		SortFunc: SortByFitnessAndNeurons,
	}
}

func getGenerationalModelTournament(n uint) eaopt.Model {
	return eaopt.ModGenerational{
		Selector:  eaopt.SelTournament{NContestants: n},
		MutRate:   mutProb,
		CrossRate: crossProb,

		ExtraOperators: []eaopt.ExtraOperator{
			eaopt.ExtraOperator{Operator: AddNeuron, Probability: addNeuronProb},
			eaopt.ExtraOperator{Operator: RemoveNeuron, Probability: removeNeuronProb},
			eaopt.ExtraOperator{Operator: SubstituteNeuron, Probability: substituteNeuronProb},
			eaopt.ExtraOperator{Operator: Train, Probability: trainProb},
		},

		SortFunc: SortByFitnessAndNeurons,
	}
}

func getGenerationalModelRoulette() eaopt.Model {
	return eaopt.ModGenerational{
		Selector:  eaopt.SelRoulette{},
		MutRate:   mutProb,
		CrossRate: crossProb,

		ExtraOperators: []eaopt.ExtraOperator{
			eaopt.ExtraOperator{Operator: AddNeuron, Probability: addNeuronProb},
			eaopt.ExtraOperator{Operator: RemoveNeuron, Probability: removeNeuronProb},
			eaopt.ExtraOperator{Operator: SubstituteNeuron, Probability: substituteNeuronProb},
			eaopt.ExtraOperator{Operator: Train, Probability: trainProb},
		},

		SortFunc: SortByFitnessAndNeurons,
	}
}
