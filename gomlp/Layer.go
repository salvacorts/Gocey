package gomlp

// Layer is a struct containing Neurons
type Layer struct {
	Neurons []Neuron
}

// MakeLayer creates a layer of `size` neurons
func MakeLayer(size int, previousLayerSize int) Layer {
	layer := Layer{make([]Neuron, size)}

	for i := 0; i < len(layer.Neurons); i++ {
		layer.Neurons[i] = MakeNeuron(previousLayerSize)
	}

	return layer
}
