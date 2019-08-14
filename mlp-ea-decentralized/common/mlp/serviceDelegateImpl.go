package mlp

import (
	"github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp/common/utils"
	"github.com/salvacorts/eaopt"
)

// DelegateImpl Implements the ga.ServiceDelegate
type DelegateImpl struct{}

// SerializeProblemDescription serializes the current config to a []byte
func (d DelegateImpl) SerializeProblemDescription() []byte {
	desc := MLPDescription{
		Epochs:       int64(Config.Epochs),
		Folds:        int64(Config.Folds),
		TrainDataset: utils.PatternsToCSV(Config.TrainingSet),
		Classes:      Config.Classes,
	}

	buff, err := desc.Marshal()
	if err != nil {
		Log.Fatalf("Could not serialize MLPDespription. %s", err.Error())
	}

	return buff
}

// DeserializeProblemDescription sets mlp.Config from deserializing buff
func (d DelegateImpl) DeserializeProblemDescription(buff []byte) {
	desc := MLPDescription{}

	err := desc.Unmarshal(buff)
	if err != nil {
		Log.Fatalf("Could not deserialize MLPDespription. %s", err.Error())
	}

	Config.Epochs = int(desc.Epochs)
	Config.Folds = int(desc.Folds)
	Config.Classes = desc.Classes

	Config.TrainingSet, err, _ = utils.LoadPatternsFromCSV(desc.TrainDataset)
	if err != nil {
		Log.Fatalf("Could not Parse patterns from reterived CSV. %s", err.Error())
	}
}

// SerializeGenome serializes a MLP to a []byte
func (d DelegateImpl) SerializeGenome(genome eaopt.Genome) []byte {
	mlp := genome.(*MultiLayerNetwork)

	buff, err := mlp.Marshal()
	if err != nil {
		Log.Fatalf("Could not serialize MultiLayerNetwork. %s", err.Error())
	}

	return buff
}

// DeserializeGenome deserialized buff to a MLP
func (d DelegateImpl) DeserializeGenome(buff []byte) eaopt.Genome {
	mlp := &MultiLayerNetwork{}

	err := mlp.Unmarshal(buff)
	if err != nil {
		Log.Fatalf("Could not deserialize MultiLayerNetwork. %s", err.Error())
	}

	return mlp
}
