package ga

import (
	"math"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/sirupsen/logrus"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// MetricsServer is a prometheus server
type MetricsServer struct {
	log  *logrus.Logger
	addr string
}

// MakeMetricsServer starts a metric server listening on addr
func MakeMetricsServer(pool *PoolModel, addr string, logger *logrus.Logger) *MetricsServer {
	ms := &MetricsServer{
		log:  logger,
		addr: addr,
	}

	// Total number of evaluations
	if err := prometheus.Register(prometheus.NewCounterFunc(
		prometheus.CounterOpts{
			Name: "island_total_evaluations",
			Help: "Number of evaluations carried so far over the population on this island",
		},
		func() float64 {
			return float64(pool.GetTotalEvaluations())
		},
	)); err != nil {
		ms.log.Fatalf("Could not start island_total_evaluations collector, %s", err.Error())
	}

	// Population size
	if err := prometheus.Register(prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "island_population_size",
			Help: "Number of individuals in the population at a given moment",
		},
		func() float64 {
			return float64(pool.population.Length())
		},
	)); err != nil {
		ms.log.Fatalf("Could not start island_population_size collector, %s", err.Error())
	}

	// Best Fitness
	if err := prometheus.Register(prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "best_fitness",
			Help: "Fitness of the best solution found so far",
		},
		func() float64 {
			return pool.BestSolution.Fitness
		},
	)); err != nil {
		ms.log.Fatalf("Could not start best_fitness collector, %s", err.Error())
	}

	// Best neurons (Delegate to problem)

	// Avg fitness
	if err := prometheus.Register(prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "population_avg_fitness",
			Help: "Average fitness of the population",
		},
		func() float64 {
			return pool.GetAverageFitness()
		},
	)); err != nil {
		ms.log.Fatalf("Could not start population_avg_fitness collector, %s", err.Error())
	}

	// Std fitness
	if err := prometheus.Register(prometheus.NewGaugeFunc(
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
	)); err != nil {
		ms.log.Fatalf("Could not start population_std_fitness collector, %s", err.Error())
	}

	// Cluster number of nodes
	if err := prometheus.Register(prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "cluster_size",
			Help: "Nodes that this node knows about",
		},
		func() float64 {
			return float64(pool.cluster.GetNumNodes())
		},
	)); err != nil {
		ms.log.Fatalf("Could not start cluster_size collector, %s", err.Error())
	}

	return ms
}

// Start the metrics server
func (ms *MetricsServer) Start() {
	ms.log.Infof("Metrics listening on: http://%s/metrics", ms.addr)

	http.Handle("/metrics", promhttp.Handler())
	err := http.ListenAndServe(ms.addr, nil)
	if err != nil {
		ms.log.Fatalf("Could not start listener. %s", err.Error())
	}
}
