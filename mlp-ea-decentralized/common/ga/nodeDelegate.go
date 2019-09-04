//+build !js

package ga

import (
	"github.com/hashicorp/memberlist"
	"github.com/sirupsen/logrus"
)

type nodeDelegate struct {
	Metadata     NodeMetadata
	Logger       *logrus.Logger
	NewIndivChan chan Individual

	broadcasts *memberlist.TransmitLimitedQueue
}

func makeNodeDelegate(log *logrus.Logger, newIndivChan chan Individual, queue *memberlist.TransmitLimitedQueue) *nodeDelegate {
	return &nodeDelegate{
		Logger:       log,
		NewIndivChan: newIndivChan,
		broadcasts:   queue,
	}
}

func (d *nodeDelegate) Broadcast(indiv Individual) {
	d.Logger.Debugf("Broadcasting indivd: [%s - %.2f]",
		indiv.IndividualID, indiv.Fitness)
	d.broadcasts.QueueBroadcast(indiv)
}

// Node meta implements membership.Delegate interface
func (d *nodeDelegate) NodeMeta(limit int) []byte {
	buff, err := d.Metadata.Marshal()
	if err != nil {
		d.Logger.Errorf("Could not serialized node metadata. %s", err.Error())
	} else if len(buff) > limit {
		d.Logger.Errorf("Metadata Serialized size (%d) exceed the limit (%d).", len(buff), limit)
	}

	return buff
}

func (d *nodeDelegate) NotifyMsg(msg []byte) {
	indiv := Individual{}
	err := indiv.Unmarshal(msg)
	if err != nil {
		d.Logger.Errorf("Could not deserialize gossiped Individual. %s", err.Error())
	}

	d.Logger.Debugf("Received broadcast about [%s - %.2f]",
		indiv.IndividualID, indiv.Fitness)

	d.NewIndivChan <- indiv
}

func (d *nodeDelegate) GetBroadcasts(overhead, limit int) [][]byte {
	return d.broadcasts.GetBroadcasts(overhead, limit)
}

func (d *nodeDelegate) LocalState(join bool) []byte {
	d.Logger.Debug("On LocalState. NOP")
	return []byte("")
}

func (d *nodeDelegate) MergeRemoteState(buf []byte, join bool) {
	d.Logger.Debug("On MergeRemoteState. NOP")
}
