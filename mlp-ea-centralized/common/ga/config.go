package ga

const (
	// Precission states the fitness precission.
	// e.g. 100 is 2 decimals (default)
	Precission float64 = 100

	// TrainEpochs states how many epochs to train the MLP in the Train operator
	TrainEpochs int = 100

	// MutateRate states the probability of a neuron to be mutated once the Mutate operator is applied
	MutateRate float64 = 1

	generations int = 100
	popSize     int = 100

	mutProb              float64 = 0.3
	crossProb            float64 = 0.3
	addNeuronProb        float64 = 0.3
	removeNeuronProb     float64 = 0.15
	substituteNeuronProb float64 = 0.15
	trainProb            float64 = 0.3
)
