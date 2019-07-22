package common

import "math/rand"

// RandomNeuronInit initialize neuron weight, bias and learning rate using NormFloat64 random value.
func RandomNeuronInit(neuron *NeuronUnit, dim int) {

	neuron.Weights = make([]float64, dim)

	// init random weights
	for index := range neuron.Weights {
		// init random threshold weight
		neuron.Weights[index] = rand.NormFloat64() * ScalingFactor
	}

	// init random bias and lrate
	neuron.Bias = rand.NormFloat64() * ScalingFactor
	neuron.Lrate = rand.NormFloat64() * ScalingFactor
	neuron.Value = rand.NormFloat64() * ScalingFactor
	neuron.Delta = rand.NormFloat64() * ScalingFactor
}

// PrepareLayer create a NeuralLayer with n NeuronUnits inside
// [n:int] is an int that specifies the number of neurons in the NeuralLayer
// [p:int] is an int that specifies the number of neurons in the previous NeuralLayer
// It returns a NeuralLayer object
func PrepareLayer(n int, p int) (l NeuralLayer) {

	l = NeuralLayer{NeuronUnits: make([]NeuronUnit, n), Length: int64(n)}

	for i := 0; i < n; i++ {
		RandomNeuronInit(&l.NeuronUnits[i], p)
	}

	return
}

// PrepareMLPNet create a multi layer Perceptron neural network.
// [l:[]int] is an int array with layers neurons number [input, ..., output]
// [lr:int] is the learning rate of neural network
// [tr:transferFunction] is a transfer function
// [tr:transferFunction] the respective transfer function derivative
func PrepareMLPNet(l []int, lr float64, tf TransferFunc) (mlp MultiLayerNetwork) {
	mlp.LRate = lr
	mlp.TFunc = tf

	mlp.NeuralLayers = make([]NeuralLayer, len(l))

	// for each layers specified
	for il, ql := range l {

		// if it is not the first
		if il != 0 {

			// prepare the GENERIC layer with specific dimension and correct number of links for each NeuronUnits
			mlp.NeuralLayers[il] = PrepareLayer(ql, l[il-1])

		} else {

			// prepare the INPUT layer with specific dimension and No links to previous.
			mlp.NeuralLayers[il] = PrepareLayer(ql, 0)

		}

	}

	return
}
