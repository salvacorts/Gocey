package ga

import "github.com/salvacorts/eaopt"

// ServiceDelegate must be implemented by the client and server function
type ServiceDelegate interface {
	// Write serialized problem description
	SerializeProblemDescription() []byte

	// Deserialize problem description into own scope
	DeserializeProblemDescription([]byte)

	// Serialize Genome to array
	SerializeGenome(eaopt.Genome) []byte

	// Deserialize array to Genome
	DeserializeGenome([]byte) eaopt.Genome
}
