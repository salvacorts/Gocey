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
	Logger                  *logrus.Logger
	ListeningPort           int
	BoostrapPeers           []string
	BroadcastBestIndividual chan Individual
	ReceiveBestIndividual   chan Individual
	list                    *memberlist.Memberlist
	delegate                *nodeDelegate
	stop                    chan bool
}

type eventHandler struct {
	Logger *logrus.Logger
}

// MakeCluster creates a cluster with default logger and listening on :9999
func MakeCluster(listenPort int, buffSize int, newBestIndividualChan chan Individual, boostrapPeers []string, logger ...*logrus.Logger) *Cluster {
	c := &Cluster{
		Logger:                  logrus.StandardLogger(),
		ListeningPort:           listenPort,
		BoostrapPeers:           boostrapPeers,
		BroadcastBestIndividual: make(chan Individual, buffSize),
		ReceiveBestIndividual:   newBestIndividualChan,
		stop:                    make(chan bool),
	}

	// If a logger is provided, set it instead of the logrus standard one
	if len(logger) > 0 {
		c.Logger = logger[0]
	}

	return c
}

// PrintMembers of the p2p cluster
func (c *Cluster) PrintMembers() {
	for {
		select {
		default:
			time.Sleep(5 * time.Second)

			nodes := c.list.Members()
			str := fmt.Sprintf("Nodes - %d:\n", len(nodes))

			for _, node := range nodes {
				str += fmt.Sprintf("%s:%d - %s\n", node.Addr.String(), node.Port, node.Name)
			}

			c.Logger.Info(str)

		case <-c.stop:
			return
		}

	}
}

// Shutdown the cluster node
func (c *Cluster) Shutdown() {
	c.Logger.Info("Shutting down cluster")
	close(c.stop)
}

// Start creates a new node with metadata for this process and joins a existing cluster
// by connecting to peers in cluster.boostrapPeers array
func (c *Cluster) Start(metadata NodeMetadata) {
	// Default conf for wide networks with event listener
	conf := memberlist.DefaultWANConfig()
	conf.Events = &eventHandler{c.Logger}
	conf.BindPort = c.ListeningPort
	conf.AdvertisePort = conf.BindPort
	conf.Name = uuid.New().String()

	queue := new(memberlist.TransmitLimitedQueue)
	queue.NumNodes = c.GetNumNodes
	conf.UDPBufferSize = 65535
	conf.Delegate = makeNodeDelegate(c.Logger, c.ReceiveBestIndividual, queue)

	c.delegate = conf.Delegate.(*nodeDelegate)
	c.delegate.Metadata = metadata

	// Create a new Node
	list, err := memberlist.Create(conf)
	if err != nil {
		c.Logger.Fatalf("Failed to create memberlist. %s", err.Error())
	}

	c.list = list
	c.Logger.Infof("Local cluster node: %s", c.list.LocalNode().String())
	defer c.list.Shutdown()

	// Try to join a cluster with bootstap nodes.
	// If it fails, be an standalone node waiting for incoming connections
	c.Logger.Debugf("Boostrap peers: %v", c.BoostrapPeers)
	_, err = c.list.Join(c.BoostrapPeers)
	if err != nil {
		c.Logger.Errorf("Failed joining cluster. %s", err.Error())
		c.Logger.Warnf("Standalone node created. Should be a boostrap for another joining node")
	}

	go c.PrintMembers()

	for {
		select {
		default:
			indiv := <-c.BroadcastBestIndividual
			c.delegate.Broadcast(indiv)

		case <-c.stop:
			return
		}

	}
}

// GetMembers resturns the members on the cluster
func (c *Cluster) GetMembers() []*memberlist.Node {
	return c.list.Members()
}

// GetNumNodes resturns the number of nodes in the cluster
func (c *Cluster) GetNumNodes() int {
	return c.list.NumMembers()
}

func (c *eventHandler) NotifyJoin(node *memberlist.Node) {
	meta := NodeMetadata{}
	meta.Unmarshal(node.Meta)

	c.Logger.Infof("Node joined: %s:%d - Grpc: (%d, %d)",
		node.Addr.String(), node.Port, meta.GrpcPort, meta.GrpcWsPort)
}

func (c *eventHandler) NotifyLeave(node *memberlist.Node) {
	c.Logger.Infof("Node left: %s:%d", node.Addr.String(), node.Port)
}

func (c *eventHandler) NotifyUpdate(node *memberlist.Node) {
	meta := NodeMetadata{}
	err := meta.Unmarshal(node.Meta)
	if err != nil {
		c.Logger.Errorf("Could not deserialize node metadata. %s", err.Error())
	}

	c.Logger.Infof("Node updated: %s:%d", node.Addr.String(), node.Port)
}
