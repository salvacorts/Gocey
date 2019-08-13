package ga

import (
	"os"
	"os/signal"
	"syscall"
	"time"

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

// Join creates a new node for this process and joins a existing cluster
// by connecting to peers in cluster.boostrapPeers array
func (c *Cluster) Join() {
	// Default conf for wide networks with event listener
	conf := memberlist.DefaultWANConfig()
	conf.Events = &eventHandler{c.Logger}
	conf.BindPort = c.ListeningPort

	// Create a new Node
	list, err := memberlist.Create(conf)
	if err != nil {
		c.Logger.Fatalf("Failed to create memberlist. %s", err.Error())
	}

	c.list = list
	c.Logger.Infof("Local cluster node: %s", c.list.LocalNode().String())

	// Try to join a cluster with bootstap nodes.
	// If it fails, be an standalone node waiting for incoming connections
	_, err = c.list.Join(c.BoostrapPeers)
	if err != nil {
		c.Logger.Errorf("Failed joining cluster. %s", err.Error())
		c.Logger.Warnf("Standalone node created. Should be a boostrap for another joining node")
	}

	// Handle exit signals to gracefully leave the cluster
	// If force stopped, the gossip protocol will take care of node failure
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP)

	<-stop

	c.Logger.Info("Leaving cluster. Timeout set to 5 seconds")
	c.list.Leave(5 * time.Second)
}

// GetMembers resturns the members on the cluster
func (c *Cluster) GetMembers() []*memberlist.Node {
	return c.list.Members()
}

func (c *eventHandler) NotifyJoin(node *memberlist.Node) {
	c.Logger.Infof("Node joined: %s", node.String())
}

func (c *eventHandler) NotifyLeave(node *memberlist.Node) {
	c.Logger.Infof("Node lest: %s", node.String())
}

func (c *eventHandler) NotifyUpdate(node *memberlist.Node) {
	c.Logger.Infof("Node updated: %s", node.String())
}
