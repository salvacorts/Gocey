package common

import (
	"errors"
	"math/rand"

	"github.com/patrickmn/go-cache"

	"github.com/salvacorts/eaopt"
)

// Population stores individuals
type Population struct {
	cache *cache.Cache
}

// MakePopulation returns an empty population
func MakePopulation() *Population {
	return &Population{
		cache: cache.New(cache.NoExpiration, cache.NoExpiration),
	}
}

// Add an individual
func (pop *Population) Add(indiv eaopt.Individual) {
	pop.cache.Set(indiv.ID, indiv, cache.NoExpiration)
}

// Remove an individual
func (pop *Population) Remove(indiv eaopt.Individual) {
	pop.cache.Delete(indiv.ID)
}

// Length returns the size of the population
func (pop *Population) Length() int {
	return pop.cache.ItemCount()
}

// Fold applies function f to each individual in the population
// until the end is reached or f returns false
func (pop *Population) Fold(f func(eaopt.Individual) bool) {
	items := pop.cache.Items()

	for _, item := range items {
		cont := f(item.Object.(eaopt.Individual))

		if !cont {
			break
		}
	}
}

// Snapshot returns the state of the population
// at the moment it was called.
// TODO: Wrap the cache.Item type to eaopt.Individual
func (pop *Population) Snapshot() map[string]cache.Item {
	return pop.cache.Items()
}

// RandIndividual takes an snapshot of the population and returns a random entry from it
func (pop *Population) RandIndividual(rnd *rand.Rand) (eaopt.Individual, error) {
	items := pop.cache.Items()

	target := rnd.Intn(len(items))
	idx := 0

	for _, item := range items {
		if idx == target {
			return item.Object.(eaopt.Individual), nil
		}

		idx++
	}

	return eaopt.Individual{}, errors.New("Could not get random individual")
}
