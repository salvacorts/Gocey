package common

import (
	mn "github.com/made2591/go-perceptron-go/model/neural"
	"github.com/salvacorts/eaopt"
)

type neuralLayerSlice mn.NeuralLayer

func (nl neuralLayerSlice) At(i int) interface{} {
	return nl.NeuronUnits[i]
}

func (nl neuralLayerSlice) Set(i int, v interface{}) {
	nl.NeuronUnits[i] = v.(mn.NeuronUnit)
}

func (nl neuralLayerSlice) Len() int {
	return nl.Length
}

func (nl neuralLayerSlice) Swap(i, j int) {
	nl.NeuronUnits[i], nl.NeuronUnits[j] = nl.NeuronUnits[j], nl.NeuronUnits[i]
}

func (nl neuralLayerSlice) Slice(a, b int) eaopt.Slice {
	return neuralLayerSlice{
		Length:      b - a,
		NeuronUnits: nl.NeuronUnits[a:b],
	}
}

func (nl neuralLayerSlice) Split(k int) (eaopt.Slice, eaopt.Slice) {
	s1 := neuralLayerSlice{
		Length:      k,
		NeuronUnits: nl.NeuronUnits[:k],
	}

	s2 := neuralLayerSlice{
		Length:      nl.Length - k,
		NeuronUnits: nl.NeuronUnits[k:],
	}

	return s1, s2
}

func (nl neuralLayerSlice) Append(s eaopt.Slice) eaopt.Slice {
	return neuralLayerSlice{
		Length:      nl.Length + s.Len(),
		NeuronUnits: append(nl.NeuronUnits, s.(neuralLayerSlice).NeuronUnits...),
	}
}

func (nl neuralLayerSlice) Replace(s eaopt.Slice) {
	nl.Length = s.Len()
	nl.NeuronUnits = s.(neuralLayerSlice).NeuronUnits
}

func (nl neuralLayerSlice) Copy() eaopt.Slice {
	s := neuralLayerSlice{
		Length:      nl.Length,
		NeuronUnits: append([]mn.NeuronUnit{}, nl.NeuronUnits...),
	}

	for i := range s.NeuronUnits {
		s.NeuronUnits[i].Weights =
			append([]float64{}, nl.NeuronUnits[i].Weights...)
	}

	return s
}
