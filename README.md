[![Build Status](https://travis-ci.org/salvacorts/TFG-Parasitic-Metaheuristics.svg?branch=master)](https://travis-ci.org/salvacorts/TFG-Parasitic-Metaheuristics)

# Bacherlor Thesis: *Gocey. Distributed Evolutionary Algorithms on Ephemeral Infrastructure*

### **Author:** Salvador Corts Sánchez
### **Supervisor:** Juan Julián Merelo Guervós

Thesis Document: https://github.com/salvacorts/TFG-Thesis
___

## Requirements

- Go >= 1.12: https://golang.org/doc/install
- Go Protocol Buffers: https://github.com/golang/protobuf

## Go dependecies

```bash
go get github.com/salvacorts/TFG-Parasitic-Metaheuristics
cd $GOPATH/src/github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized/native
go get ./...
```

## Run

All the following paths are relative to `$GOPATH/src/github.com/salvacorts/TFG-Parasitic-Metaheuristics/mlp-ea-decentralized` directory.

#### Island

```bash
cd native/server
go run -grpcPort <port> -clusterPort <port> -metricsPort <port> -clusterBoostrap <boostrap node addr:port> -datasetPath <path to dataset> -webPath <path to web> -webPort <port>
# e.g. go run server.go -grpcPort 2006 -clusterPort 4003 -metricsPort 5003 -clusterBoostrap 127.0.0.1:4001 -datasetPath ../../../datasets/glass.csv -webpath ../../web/src/ -webPort 8080
```

#### Evaluator

##### Native

```bash
cd native/client
go run client.go -server <island addr:port>
# e.g. go run client.go -server 127.0.0.1:2006
```

##### Browser (WebAssembly)
Open a new tab in yout browser and enter an island IP and its *webPort* port. E.g. if an island is running at 127.0.0.1, and its *webPort* parameter is 8080, go to [127.0.0.1:8080](127.0.0.1:8080).

**Modifying Go Browser Client Source:** You will need to rebuild the webassembly file
```bash
cd web/src/go
GOOS=js GOARCH=wasm go build -o wasm/main.wasm
```
