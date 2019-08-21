package ga

import (
	"math"
	"math/rand"
	"testing"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
)

func TestMakePool(t *testing.T) {
	size := 100

	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := MakePool(size, 9999, 9998, []string{}, rand.New(rand.NewSource(7)), mlp.NewRandMLP)

	if pool.GetTotalEvaluations() != 0 {
		t.Errorf("Evaluated individuals on initialization")
	}

	if pool.BestSolution.Fitness != math.Inf(1) {
		t.Errorf("Best individual fitness is not +Inf")
	}

	if pool.PopSize != size {
		t.Errorf("PopSize and size argument on costructor do noot match")
	}

	if pool.population.Length() != size {
		t.Errorf("Population Length does not match the constructor size")
	}

	if len(pool.GetPopulationSnapshot()) != pool.PopSize {
		t.Errorf("Population snapshot size is different form PopSize")
	}
}

func TestSelection(t *testing.T) {
	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := MakePool(20, 9999, 9998, []string{}, rand.New(rand.NewSource(7)), mlp.NewRandMLP)
	pool.SortFunc = mlp.SortByFitnessAndNeurons

	for i, in := range pool.GetPopulationSnapshot() {
		in.Fitness = float64(i)
		pool.population.Add(in)
	}

	if pool.population.Length() != 20 {
		t.Errorf("Population Add operator do not overwrite existing individual")
	}

	selected := pool.selection(4, 4)

	scores := make([]float64, len(selected))
	for i, s := range selected {
		scores[i] = s.Fitness
	}

	t.Logf("Selected individuals: %v", scores)
	if len(selected) != 4 {
		t.Errorf("Selection do not select the amount of offsprings specified")
	}

	// If selection selected one of the 4 worst individuals it is not working
	// It can happen but it is very unlikely
	for _, s := range selected {
		for i := 20 - 4 - 1; i < 20; i++ {
			if s.Fitness == float64(i) {
				t.Log("[WARN] Selected one of the worst individuals. It is very unlikely")
			}
		}
	}

}

func TestCrossover(t *testing.T) {
	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := MakePool(20, 9999, 9998, []string{}, rand.New(rand.NewSource(7)), mlp.NewRandMLP)
	pool.SortFunc = mlp.SortByFitnessAndNeurons
	pool.CrossRate = 1

	snap := pool.GetPopulationSnapshot()

	offsprings := pool.crossover(snap[:4])

	for i, o := range offsprings {
		if o == snap[i] {
			t.Errorf("Offspring is equal to parent")
		}
	}
}

func TestGetAverageFitness(t *testing.T) {
	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := MakePool(20, 9999, 9998, []string{}, rand.New(rand.NewSource(7)), mlp.NewRandMLP)

	sum := 0.0
	for i, in := range pool.GetPopulationSnapshot() {
		in.Fitness = float64(i)
		pool.population.Add(in)

		sum += in.Fitness
	}

	avg := pool.GetAverageFitness()
	if avg != sum/20 {
		t.Errorf("Average fitness is not correct. Expected %f, got %f", sum/20, avg)
	}

}
