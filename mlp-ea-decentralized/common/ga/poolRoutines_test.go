package ga

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/salvacorts/eaopt"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
)

func TestHandleEvaluate(t *testing.T) {
	size := 10

	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := MakePool(size, 9999, 9998, 9997, []string{}, rand.New(rand.NewSource(7)), mlp.NewRandMLP)
	defer pool.metricsServer.Shutdown()

	go pool.handleEvaluate()

	// Wait for GRPC server to wake
	time.Sleep(2 * time.Second)

	services := pool.grpcServer.GetServiceInfo()
	if len(services) != 1 {
		t.Errorf("ga.DistributedEA service not started")
	}

	close(pool.stop)
}

func TestHandleEvaluated(t *testing.T) {
	size := 10

	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := MakePool(size, 9999, 9998, 9997, []string{}, rand.New(rand.NewSource(7)), mlp.NewRandMLP)
	defer pool.metricsServer.Shutdown()
	pool.MaxEvaluations = 100
	pool.Delegate = mlp.DelegateImpl{}

	pool.BestCallback = func(p *PoolModel) {
		t.Logf("New best solution successfully updated")
	}

	// All individuals to fitness = 100.0
	for _, in := range pool.GetPopulationSnapshot() {
		in.Fitness = 100.0
		in.Evaluated = true
		pool.population.Add(in)
	}

	go pool.handleEvaluated()

	// Wait for goroutine to start
	time.Sleep(1 * time.Second)

	pool.evaluatedChannel <- eaopt.Individual{
		ID:        "Test",
		Fitness:   7,
		Evaluated: true,
		Genome:    mlp.NewRandMLP(rand.New(rand.NewSource(7))),
	}

	// Wait for the goroutine to process the test individual
	time.Sleep(1 * time.Second)

	if pool.BestSolution.ID != "Test" ||
		pool.BestSolution.Fitness != 7 {
		t.Errorf("Best solution not updated")
	}

	close(pool.stop)
}

func TestHandleBroadcastedIndividual(t *testing.T) {
	size := 10

	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := MakePool(size, 9999, 9998, 9997, []string{}, rand.New(rand.NewSource(7)), mlp.NewRandMLP)
	defer pool.metricsServer.Shutdown()
	pool.MaxEvaluations = 100
	pool.Delegate = mlp.DelegateImpl{}

	pool.BestCallback = func(p *PoolModel) {
		t.Logf("New best solution successfully updated")
	}

	// All individuals to fitness = 100.0
	for _, in := range pool.GetPopulationSnapshot() {
		in.Fitness = 100.0
		in.Evaluated = true
		pool.population.Add(in)
	}

	go pool.handleBroadcastedIndividual()

	// Wait for goroutine to start
	time.Sleep(1 * time.Second)

	pool.BroadcastedIndividual <- Individual{
		IndividualID: "Test_Best",
		Fitness:      7,
		Evaluated:    true,
		Genome: pool.Delegate.SerializeGenome(
			mlp.NewRandMLP(rand.New(rand.NewSource(7)))),
	}

	// Wait for the goroutine to process the test individual
	time.Sleep(1 * time.Second)

	if pool.BestSolution.ID != "Test_Best" ||
		pool.BestSolution.Fitness != 7 {
		t.Errorf("Best solution not updated")
	}

	pool.BroadcastedIndividual <- Individual{
		IndividualID: "Test_Worst",
		Fitness:      8,
		Evaluated:    true,
		Genome: pool.Delegate.SerializeGenome(
			mlp.NewRandMLP(rand.New(rand.NewSource(7)))),
	}

	// Wait for the goroutine to process the test individual
	time.Sleep(1 * time.Second)

	if pool.BestSolution.ID == "Test_Worst" ||
		pool.BestSolution.Fitness == 8 {
		t.Errorf("Best solution updated to worst solution")
	}

	close(pool.stop)
}

func TestPopulationGrowControl(t *testing.T) {
	size := 100

	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := MakePool(size, 9999, 9998, 9997, []string{}, rand.New(rand.NewSource(7)), mlp.NewRandMLP)
	defer pool.metricsServer.Shutdown()
	pool.SortFunc = mlp.SortByFitnessAndNeurons

	// All individuals to fitness = 1.0
	for _, in := range pool.GetPopulationSnapshot() {
		in.Fitness = 1.0
		pool.population.Add(in)
	}

	// Also the best one
	pool.BestSolution.Fitness = 0.7
	pool.BestSolution.ID = "Best"

	if pool.population.Length() != size {
		t.Errorf("Population Add operator do not overwrite existing individual")
	}

	// Add individuals surplus all with Fitness > than original ones
	for i := 0; i < 100; i++ {
		pool.population.Add(eaopt.Individual{
			ID:        fmt.Sprintf("Indiv-%d", i),
			Fitness:   float64(7 + i),
			Evaluated: false,
			Genome:    mlp.NewRandMLP(rand.New(rand.NewSource(7))),
		})
	}

	if pool.population.Length() != pool.PopSize+100 {
		t.Errorf("Failed to add individuals for test into the population")
	}

	go pool.populationGrowControl()

	// Wait for goroutine to start and complete
	time.Sleep(3 * time.Second)

	if pool.population.Length() != pool.PopSize {
		t.Errorf("Goroutine has not deleted the individuals surplus")
	}

	bestFound := false
	for _, in := range pool.GetPopulationSnapshot() {
		if in.Fitness > 1.0 {
			t.Errorf("GrowthControl removed individuals with better fitness: %f", in.Fitness)
		}

		if in.Fitness == pool.BestSolution.Fitness {
			bestFound = true
		}
	}

	if !bestFound {
		t.Errorf("Worst of the remaining population was not replaced by a copy of the best solution so far")
	}

	close(pool.stop)
}
