package ga

import (
	"testing"

	"github.com/salvacorts/eaopt"
)

func TestMake(t *testing.T) {
	population := MakePopulation()

	if population.Length() > 0 {
		t.Errorf("Population is not empty at initialization")
	}
}

func TestAdd(t *testing.T) {
	population := MakePopulation()

	indiv := eaopt.Individual{
		ID:        "A",
		Fitness:   50.0,
		Evaluated: true,
	}

	population.Add(indiv)

	if population.Length() != 1 {
		t.Errorf("Individual not appended")
	}

	found := false
	snap := population.Snapshot()
	for k, v := range snap {
		if k == indiv.ID {
			found = true

			i, ok := v.Object.(eaopt.Individual)
			if !ok {
				t.Errorf("Could not cast cache object to eaopt.Individual")
			}

			if k != i.ID {
				t.Errorf("ID on individual and key do not match")
			}

			if i.Fitness != indiv.Fitness {
				t.Errorf("Content of cache individual do not match original individual")
			}

			break
		}
	}

	if !found {
		t.Errorf("Missing individual added to population")
	}
}

func TestRemove(t *testing.T) {
	population := MakePopulation()

	indiv := eaopt.Individual{
		ID:        "A",
		Fitness:   50.0,
		Evaluated: true,
	}

	population.Add(indiv)
	if population.Length() != 1 {
		t.Errorf("Individual not appended")
	}

	population.Remove(indiv)

	snap := population.Snapshot()
	for k := range snap {
		if k == indiv.ID {
			t.Errorf("Individual was not removed form population")
			break
		}
	}
}

func TestFold(t *testing.T) {
	population := MakePopulation()

	indiv1 := eaopt.Individual{
		ID:        "A",
		Fitness:   50.0,
		Evaluated: true,
	}

	indiv2 := eaopt.Individual{
		ID:        "B",
		Fitness:   60.0,
		Evaluated: true,
	}

	population.Add(indiv1)
	population.Add(indiv2)
	if population.Length() != 2 {
		t.Errorf("Individuals not appended")
	}

	population.Fold(func(i eaopt.Individual) bool {
		i.Fitness += 10.0
		return true
	})

	snap := population.Snapshot()

	if snap[indiv1.ID].Object.(eaopt.Individual).Fitness != indiv1.Fitness {
		t.Errorf("Individual 1 was modified after fold")
	}

	if snap[indiv2.ID].Object.(eaopt.Individual).Fitness != indiv2.Fitness {
		t.Errorf("Individual 2 was modified after fold")
	}
}
