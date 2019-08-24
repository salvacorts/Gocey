package ga

import (
	"math"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/sirupsen/logrus"

	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// MetricsServer is a prometheus server
type MetricsServer struct {
	log  *logrus.Logger
	addr string

	EvaluationsCount        prometheus.Counter
	OutgoingMigrationsCount prometheus.Counter
	IncomingMigrationsCount prometheus.Counter
	OutgoingBroadcasts      prometheus.Counter
	IncomingBroadcasts      prometheus.Counter
	BestFitnessGauge        prometheus.Gauge

	popSizeCollector     prometheus.Collector
	avgFitnessCollector  prometheus.Collector
	stdFitnessCollector  prometheus.Collector
	clusterSizeCollector prometheus.Collector
}

// MakeMetricsServer starts a metric server listening on addr
func MakeMetricsServer(pool *PoolModel, addr string, logger *logrus.Logger) *MetricsServer {
	ms := &MetricsServer{
		log:  logger,
		addr: addr,
	}

	// Total number of evaluations
	ms.EvaluationsCount = promauto.NewCounter(prometheus.CounterOpts{
		Name: "island_total_evaluations",
		Help: "Number of evaluations carried out so far over the population on this island",
	})

	// Best Fitness
	ms.BestFitnessGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "best_fitness",
		Help: "Fitness of the best solution found so far",
	})

	// MigrationsCount
	ms.OutgoingMigrationsCount = promauto.NewCounter(prometheus.CounterOpts{
		Name: "island_total_migrations_outgoing",
		Help: "Number of migrations carried out so far",
	})

	ms.IncomingMigrationsCount = promauto.NewCounter(prometheus.CounterOpts{
		Name: "island_total_migrations_incomming",
		Help: "Number of migrations carried out so far",
	})

	// Broadcast count
	ms.OutgoingBroadcasts = promauto.NewCounter(prometheus.CounterOpts{
		Name: "island_total_broadcasts_outgoing",
		Help: "Number of broadcasts carried out so far",
	})

	ms.IncomingBroadcasts = promauto.NewCounter(prometheus.CounterOpts{
		Name: "island_total_broadcast_incomming",
		Help: "Number of broadcasts carried out so far",
	})

	// Best neurons (Delegate to problem?)

	// Population size
	ms.popSizeCollector = prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "island_population_size",
			Help: "Number of individuals in the population at a given moment",
		},
		func() float64 {
			return float64(pool.population.Length())
		},
	)

	// Avg fitness
	ms.avgFitnessCollector = prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "population_avg_fitness",
			Help: "Average fitness of the population",
		},
		func() float64 {
			return pool.GetAverageFitness()
		},
	)

	// Std fitness
	ms.stdFitnessCollector = prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "population_std_fitness",
			Help: "Standard deviation of the fitness of the population",
		},
		func() float64 {
			pop := pool.GetPopulationSnapshot()

			mean := 0.0
			for _, in := range pop {
				mean += in.Fitness
			}
			mean /= float64(len(pop))

			std := 0.0
			for _, in := range pop {
				std += math.Pow((in.Fitness - mean), 2.0)
			}
			std = math.Sqrt((1 / mean) * std)

			return std
		},
	)

	// Cluster number of nodes
	ms.clusterSizeCollector = prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "cluster_size",
			Help: "Nodes that this node knows about",
		},
		func() float64 {
			return float64(pool.cluster.GetNumNodes())
		},
	)

	return ms
}

// Start the metrics server
func (ms *MetricsServer) Start() {
	ms.log.Infof("Metrics listening on: http://%s/metrics", ms.addr)

	if err := prometheus.Register(ms.popSizeCollector); err != nil {
		ms.log.Fatalf("Could not start island_population_size collector, %s", err.Error())
	}

	if err := prometheus.Register(ms.avgFitnessCollector); err != nil {
		ms.log.Fatalf("Could not start population_avg_fitness collector, %s", err.Error())
	}

	if err := prometheus.Register(ms.stdFitnessCollector); err != nil {
		ms.log.Fatalf("Could not start population_std_fitness collector, %s", err.Error())
	}

	if err := prometheus.Register(ms.clusterSizeCollector); err != nil {
		ms.log.Fatalf("Could not start cluster_size collector, %s", err.Error())
	}

	http.Handle("/metrics", promhttp.Handler())
	err := http.ListenAndServe(ms.addr, nil)
	if err != nil {
		ms.log.Fatalf("Could not start listener. %s", err.Error())
	}
}

// Shutdown the metrics server unregistering metrics
func (ms *MetricsServer) Shutdown() {
	ms.log.Info("Shutting down metrics server")

	if ms.EvaluationsCount != nil {
		prometheus.Unregister(ms.EvaluationsCount)
	}

	if ms.OutgoingMigrationsCount != nil {
		prometheus.Unregister(ms.OutgoingMigrationsCount)
	}

	if ms.IncomingMigrationsCount != nil {
		prometheus.Unregister(ms.IncomingMigrationsCount)
	}

	if ms.OutgoingBroadcasts != nil {
		prometheus.Unregister(ms.OutgoingBroadcasts)
	}

	if ms.IncomingBroadcasts != nil {
		prometheus.Unregister(ms.IncomingBroadcasts)
	}

	if ms.BestFitnessGauge != nil {
		prometheus.Unregister(ms.BestFitnessGauge)
	}

	if ms.popSizeCollector != nil {
		prometheus.Unregister(ms.popSizeCollector)
	}

	if ms.avgFitnessCollector != nil {
		prometheus.Unregister(ms.avgFitnessCollector)
	}

	if ms.stdFitnessCollector != nil {
		prometheus.Unregister(ms.stdFitnessCollector)
	}

	if ms.clusterSizeCollector != nil {
		prometheus.Unregister(ms.clusterSizeCollector)
	}
}
