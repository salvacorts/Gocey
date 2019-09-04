//+build !js

package ga

import "github.com/hashicorp/memberlist"

// Name implements NamedBroadcast
func (in Individual) Name() string {
	return in.IndividualID
}

// Invalidates implements Broadcast interface for Individual
func (in Individual) Invalidates(other memberlist.Broadcast) bool {
	nb, ok := other.(memberlist.NamedBroadcast)
	if !ok {
		return false // since it is not of this kind
	}

	return in.Name() == nb.Name()
}

// Message implements Broadcast interface
func (in Individual) Message() []byte {
	buff, err := in.Marshal()
	if err != nil {
		Log.Fatalf("Coud not serialize Individual. %s", err.Error())
	}

	return buff
}

// Finished implements Broadcast interface
func (in Individual) Finished() {}
