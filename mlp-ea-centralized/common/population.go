package common

import (
	"errors"
	"math/rand"
	"sync"

	"github.com/MaxHalford/eaopt"
)

type Population struct {
	individuals map[string]eaopt.Individual
	mutexes     map[string]*sync.Mutex
	globalMutex *sync.Mutex

	// Todo maybe we need a global mutex for rand on map
}

func MakePopulation() *Population {
	return &Population{
		individuals: make(map[string]eaopt.Individual),
		mutexes:     make(map[string]*sync.Mutex),
		globalMutex: &sync.Mutex{},
	}
}

func (pop *Population) AddToPopulation(indiv eaopt.Individual) {
	mut := pop.mutexes[indiv.ID]
	if mut == nil {
		mut = &sync.Mutex{}
		pop.mutexes[indiv.ID] = mut
	}

	mut.Lock()
	pop.individuals[indiv.ID] = indiv
	mut.Unlock()
}

func (pop *Population) RemoveFromPopulation(indiv eaopt.Individual) {
	mut := pop.mutexes[indiv.ID]
	if mut != nil { // TODO: expect this
		Log.Fatalln("No mutex for this individual on map, does it exist?")
	}

	mut.Lock()
	delete(pop.individuals, indiv.ID)
	mut.Unlock()

	delete(pop.mutexes, indiv.ID)
}

func (pop *Population) Length() int {
	return len(pop.individuals)
}

func (pop *Population) RandIndividual(rnd *rand.Rand) (eaopt.Individual, error) {
	target := rnd.Intn(len(pop.individuals))
	idx := 0

	for _, indiv := range pop.individuals {
		if idx == target {
			return indiv, nil
		}

		idx++
	}

	return eaopt.Individual{}, errors.New("Could not get random individual")
}

// func (pop *Population) GetValues() []eaopt.Individual {
// 	values := make([]eaopt.Individual, len(pop.individuals))

// 	for _, v := range pop.individuals

// 	return values
// }
