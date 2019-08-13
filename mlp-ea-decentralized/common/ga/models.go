package ga

import (
	"math"
	"sort"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
	"github.com/salvacorts/eaopt"
)

// SortByFitnessAndNeurons First sorts by Fitness and then by number of neurons
// with a precission of n decimals (e.g. 0.01)
func SortByFitnessAndNeurons(indis []eaopt.Individual, precission float64) {

	// Round with precission
	round := func(i float64) float64 {
		return math.Floor(i*precission) / precission
	}

	// Sort individuals first by fitness and then by number of neurons
	sort.Slice(indis, func(i, j int) bool {

		if round(indis[i].Fitness) == round(indis[j].Fitness) {
			neuronsI := 0
			neuronsJ := 0

			for _, l := range indis[i].Genome.(*mlp.MLP).NeuralLayers {
				neuronsI += int(l.Length)
			}

			for _, l := range indis[j].Genome.(*mlp.MLP).NeuralLayers {
				neuronsJ += int(l.Length)
			}

			return neuronsI < neuronsJ
		}

		return indis[i].Fitness < indis[j].Fitness
	})
}
