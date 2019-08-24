package ga

import (
	"context"
	"math/rand"
	"testing"

	"github.com/golang/protobuf/ptypes/empty"

	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/common/mlp"
	"github.com/salvacorts/eaopt"
	"github.com/sirupsen/logrus"
)

func TestBorrowIndividual(t *testing.T) {
	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := &PoolModel{Delegate: mlp.DelegateImpl{}}

	// We need a buffered chan to not get stuck
	input := make(chan eaopt.Individual, 100)
	output := make(chan eaopt.Individual, 100)

	server := Server{
		Log:    logrus.New(),
		Input:  input,
		Output: output,
		Pool:   pool,
	}

	server.Input <- eaopt.Individual{
		ID:        "Test",
		Fitness:   0,
		Evaluated: false,
		Genome:    mlp.NewRandMLP(rand.New(rand.NewSource(7))),
	}

	indiv, err := server.BorrowIndividual(context.Background(), &empty.Empty{})
	if err != nil {
		t.Errorf("Error on server.BorrowIndividual(). %s", err.Error())
	} else if indiv.IndividualID != "Test" {
		t.Errorf("ID of individual do not match")
	}
}

func TestReturnIndividual(t *testing.T) {
	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := &PoolModel{Delegate: mlp.DelegateImpl{}}

	input := make(chan eaopt.Individual, 100)
	output := make(chan eaopt.Individual, 100)

	server := Server{
		Log:    logrus.New(),
		Input:  input,
		Output: output,
		Pool:   pool,
	}

	indiv := &Individual{
		IndividualID: "Test",
		Evaluated:    true,
		Fitness:      0,
		Genome: pool.Delegate.SerializeGenome(
			mlp.NewRandMLP(rand.New(rand.NewSource(7)))),
	}

	_, err := server.ReturnIndividual(context.Background(), indiv)
	if err != nil {
		t.Errorf("Error on server.ReturnIndividual(). %s", err.Error())
	} else {
		indiv2 := <-server.Output

		if indiv2.ID != indiv.IndividualID {
			t.Errorf("Returned individual ID do not match the original")
		}
	}
}

func TestMigrateIndividuals(t *testing.T) {
	size := 10
	rnd := rand.New(rand.NewSource(7))
	mlp.Config.FactoryCfg.MinHiddenNeurons = 2
	mlp.Config.FactoryCfg.MaxHiddenNeurons = 4
	pool := MakePool(size, 9999, 9998, 9997, []string{}, rnd, mlp.NewRandMLP)
	pool.Delegate = mlp.DelegateImpl{}
	pool.SortFunc = mlp.SortByFitnessAndNeurons
	pool.NMigrate = 4

	input := make(chan eaopt.Individual, 100)
	output := make(chan eaopt.Individual, 100)

	server := Server{
		Log:    logrus.New(),
		Input:  input,
		Output: output,
		Pool:   pool,
	}

	// All individuals in population to fitness [0, 10, 20, ..., 100]
	for i, in := range pool.GetPopulationSnapshot() {
		in.Fitness = float64(i * 100)
		in.Evaluated = true
		pool.population.Add(in)
	}

	batch := &IndividualsBatch{
		Individuals: []Individual{
			Individual{
				IndividualID: "Test1",
				Fitness:      5,
				Genome:       pool.Delegate.SerializeGenome(mlp.NewRandMLP(rnd)),
			},
			Individual{
				IndividualID: "Test2",
				Fitness:      15,
				Genome:       pool.Delegate.SerializeGenome(mlp.NewRandMLP(rnd)),
			},
			Individual{
				IndividualID: "Test3",
				Fitness:      1005,
				Genome:       pool.Delegate.SerializeGenome(mlp.NewRandMLP(rnd)),
			},
			Individual{
				IndividualID: "Test4",
				Fitness:      2005,
				Genome:       pool.Delegate.SerializeGenome(mlp.NewRandMLP(rnd)),
			},
		},
	}

	_, err := server.MigrateIndividuals(context.Background(), batch)
	if err != nil {
		t.Errorf("Error on server.TestMigrateIndividuals(). %s", err.Error())
	} else {

		if pool.population.Length() != size {
			t.Errorf("Migrate increased population size")
		}

		snap := pool.GetPopulationSnapshot()
		scores := make([]float64, 0, len(snap))
		found := make(map[string]bool)
		for _, is := range snap {
			scores = append(scores, is.Fitness)
			for _, ib := range batch.Individuals {
				if is.ID == ib.IndividualID {
					found[ib.IndividualID] = true
				}
			}
		}

		t.Logf("Snap: %v", scores)

		// We expect Test1 and Test2 to be on the population
		// and Test3 and T4 to not being in there
		if found["Test3"] || found["Test4"] {
			t.Errorf("Migration replaced better individuals by worse ones")
		}

		if !found["Test1"] || !found["Test2"] {
			t.Errorf("Migration did not replace better individuals by worst ones")
		}

		if pool.BestSolution.ID != "Test1" || pool.BestSolution.Fitness != 5 {
			t.Errorf("Migration did not replace the best individual")
		}
	}
}
