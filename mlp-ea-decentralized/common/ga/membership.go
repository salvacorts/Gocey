//+build !js

package ga

import (
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/hashicorp/memberlist"
	"github.com/sirupsen/logrus"
)

// Cluster represents a p2p cluster
type Cluster struct {
	Logger        *logrus.Logger
	ListeningPort int
	BoostrapPeers []string
	list          *memberlist.Memberlist
}

type eventHandler struct {
	Logger *logrus.Logger
}

// MakeCluster creates a cluster with default logger and listening on :9999
func MakeCluster() *Cluster {
	return &Cluster{
		Logger:        logrus.StandardLogger(),
		ListeningPort: 9999,
	}
}

func (c *Cluster) PrintMembers() {
	for {
		time.Sleep(5 * time.Second)

		nodes := c.list.Members()
		str := fmt.Sprintf("Nodes - %d:\n", len(nodes))

		for _, node := range nodes {
			str += fmt.Sprintf("%s:%d - %s\n", node.Addr.String(), node.Port, node.Name)
		}

		c.Logger.Info(str)
	}
}

// Join creates a new node for this process and joins a existing cluster
// by connecting to peers in cluster.boostrapPeers array
func (c *Cluster) Join(metadata NodeMeta) {
	// Default conf for wide networks with event listener
	conf := memberlist.DefaultWANConfig()
	conf.Events = &eventHandler{c.Logger}
	conf.BindPort = c.ListeningPort
	conf.Name = uuid.New().String()

	// Create a new Node
	list, err := memberlist.Create(conf)
	if err != nil {
		c.Logger.Fatalf("Failed to create memberlist. %s", err.Error())
	}

	c.list = list
	c.Logger.Infof("Local cluster node: %s", c.list.LocalNode().String())

	metaSer, err := metadata.Marshal()
	if err != nil {
		c.Logger.Fatalf("Could not serialized node metadata. %s", err.Error())
	}

	c.list.LocalNode().Meta = metaSer

	// Try to join a cluster with bootstap nodes.
	// If it fails, be an standalone node waiting for incoming connections
	_, err = c.list.Join(c.BoostrapPeers)
	if err != nil {
		c.Logger.Errorf("Failed joining cluster. %s", err.Error())
		c.Logger.Warnf("Standalone node created. Should be a boostrap for another joining node")
	}

	select {} // TODO: Replace by a stop channel
}

// GetMembers resturns the members on the cluster
func (c *Cluster) GetMembers() []*memberlist.Node {
	return c.list.Members()
}

func (c *eventHandler) NotifyJoin(node *memberlist.Node) {
	c.Logger.Infof("Node joined: %s:%d", node.Addr.String(), node.Port)
}

func (c *eventHandler) NotifyLeave(node *memberlist.Node) {
	c.Logger.Infof("Node left: %s:%d", node.Addr.String(), node.Port)
}

func (c *eventHandler) NotifyUpdate(node *memberlist.Node) {
	c.Logger.Infof("Node updated: %s:%d", node.Addr.String(), node.Port)
}
