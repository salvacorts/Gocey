package ga

import (
	"testing"
	"time"
)

func TestCluster(t *testing.T) {
	recvBestChan := make(chan Individual, 100)

	peer1 := MakeCluster(6666, 100, nil, []string{})
	peer2 := MakeCluster(6667, 100, recvBestChan, []string{"127.0.0.1:6666"})

	// Launch First peer and wait for it to be ready
	go peer1.Start(NodeMetadata{})
	time.Sleep(2 * time.Second)

	// Peer2 will connect to peer1
	go peer2.Start(NodeMetadata{})
	time.Sleep(2 * time.Second)

	if peer1.GetNumNodes() != 2 {
		t.Errorf("Peer1 could not discover peer2")
	}

	if peer2.GetNumNodes() != 2 {
		t.Errorf("Peer2 could not discover peer2")
	}

	// Test individual broadcast
	peer1.BroadcastBestIndividual <- Individual{
		IndividualID: "Test",
	}

	// Wait for gossip the individual
	time.Sleep(5 * time.Second)

	indiv := <-peer2.ReceiveBestIndividual
	t.Logf("Got indiv broadcasted with ID: %s", indiv.IndividualID)

	if indiv.IndividualID != "Test" {
		t.Errorf("Error on broadcast")
	}

	peer1.Shutdown()
	peer2.Shutdown()
}
