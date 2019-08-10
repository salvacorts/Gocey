package mlp

import (
	"github.com/salvacorts/eaopt"
)

type neurons []NeuronUnit

func (n neurons) At(i int) interface{} {
	if i > len(n)-1 {
		Log.Fatalf("Trying to At %d when Len=%d", i, len(n))
	}

	return n[i]
}

func (n neurons) Set(i int, v interface{}) {
	if i > len(n)-1 {
		Log.Fatalf("Trying to Set %d when Len=%d", i, len(n))
	}

	n[i] = v.(NeuronUnit)
}

func (n neurons) Len() int {
	return len(n)
}

func (n neurons) Swap(i, j int) {
	if i > len(n)-1 || i > len(n)-1 {
		Log.Fatalf("Trying to Swap %d:%d when Len=%d",
			i, j, len(n))
	}

	n[i], n[j] = n[j], n[i]
}

func (n neurons) Slice(a, b int) eaopt.Slice {
	if a > len(n)-1 || b > len(n) {
		Log.Fatalf("Trying to get Slice in range %d:%d when Len=%d",
			a, b, len(n))
	}

	return n[a:b]
}

func (n neurons) Split(k int) (eaopt.Slice, eaopt.Slice) {
	return n[:k], n[k:]
}

func (n neurons) Append(s eaopt.Slice) eaopt.Slice {
	return append(n, s.(neurons)...)
}

func (n neurons) Replace(s eaopt.Slice) {
	copy(n, s.(neurons))

	for i := range n {
		n[i].Weights = make([]float64, len(s.(neurons)[i].Weights))
		copy(n[i].Weights, s.(neurons)[i].Weights)
	}
}

func (n neurons) Copy() eaopt.Slice {
	clone := make(neurons, len(n))

	copy(clone, n)

	for i := range clone {
		clone[i].Weights = make([]float64, len(n[i].Weights))
		copy(clone[i].Weights, n[i].Weights)
	}

	return clone
}
