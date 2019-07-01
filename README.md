# Metaheurísticas distribuidas usando infraestructura efímera

## Introducción
Los entornos de computación modernos ofrecen todo tipo de facilidades para llevar a cabo computaciones complejas: sistemas concurrentes, distribuidos, e incluso voluntarios basados en la conexión, a través del navegador, de usuarios a un navegador.

Este tipo de entornos presentan diferentes retos, que van desde la adaptación del algoritmo a los mismos, teniendo en cuenta que los nodos pueden aparecer y desaparecer con rapidez, hasta el rediseño de algoritmo de forma que pueda trabajar en entornos concurrentes o de cloud.

En este proyecto se enfocarán diferentes tipos de metaheurísticas en diferentes entornos, comparándolos y haciendo diferentes pruebas de concepto que nos permitan averiguar qué tipo de prestaciones son esperables de los mismos

## Objetivos
El objetivo de este TFG será desarrollar un sistema colaborativo para ejecutar metaheurísticas de forma distribuidas.

Por un lado, desarrollaremos dos tipos de clientes:
- **Nativo**: Se ejecutará un binario en un nodo, ya sea este un PC o una máquina virtual en un entrono cloud.
- **Navegador**: La metaheurística se ejecutará desde una pestaña (o varias) de un navegador.

Por otro lado, se implementará un servidor capaz de manejar los nodos distribuidos independientemente del tipo de cliente. El servidor debe almacenar el código a ejecutar en el cliente tanto si este lo ejecuta nativamente como en el navegador. El sistema debe soportar la perdida espontanea de nodos.

Finalmente, se implementarán varias metaheurísticas (TBD) que ejecutar en esta plataforma a fin de comparar los resultados a los obtenidos en otras arquitecturas ya existentes.

## Posible Implementación

El sistema puede dividirse en dos partes

#### Backend
Desarrollaremos el servidor como una API en Golang. Dicha API tendrá tres funcionalidades básicas:

1. Interfaz para subir nuevos problemas y sus ejecutables.
2. Proporcionar el código a ejecutar por los nodos. Cada problema con el que colaborar puede tener un nombre como identificador asociado a uno varios ejecutables. En función del tipo de nodo (nativo o navegador) se descargará un ejecutable u otro.
     1. **Cliente Nativo:** sería interesante que el servidor devuelva un *dockerfile* donde se defina como ejecutar el programa y en que entorno.
     2. **Cliente Navegador:** El servidor redireccionará a una página que contiene el código JavaScript a ejecutar.
3. Sincronizar nodos. Por ejemplo, para un Algoritmo Genético, el servidor puede devolver la mejor solución hasta el momento cuando el nodo la solicite y ser capaz de actualizar dicha mejor solución cuando el cliente le mande su mejor solución local hasta el momento.

#### Frontend
Tanto si el cliente se ejecuta de forma nativa o en el navegador, el esquema de funcionamiento del cliente y de comunicación con el servidor debe ser el mismo.

Una cuestión importante a resolver es cada cuanto el cliente pedirá al servidor la mejor solucíon hasta el momento o comunicará su mejor solución local.

Otra cuestión a resolver es como compartir soluciones encontradas entre los nodos y el servidor ya que la representación de dischas soluciones puede variar radicalmente de problema a problema.

## Resources:

- [A modern, event-based architecture for distributed evolutionary algorithms](https://dl.acm.org/citation.cfm?id=3205719)
- [Cloudy distributed evolutionary computation](https://dl.acm.org/citation.cfm?id=3207858)
- [Mapping evolutionary algorithms to a reactive, stateless architecture: using a modern concurrent language](https://dl.acm.org/citation.cfm?id=3208317)

----------

# Setup

## Common dependencies

- Golang >= 1.11
- Fabric >= 2
- NPM

#### Optional
- goexec: `go get github.com/shurcooL/goexec`

## Native
### Dependencies
```bash
go get ./...
```

# Run
```bash
cd mlp/native
go run main.go
```


## Web
### Dependencies
```bash
cd mlp/web
npm install 
npm run build
```

# Run
```bash
cd dist
goexec 'http.ListenAndServe(":8080", http.FileServer(http.Dir(".")))'
```

Or with a Fabric:
```bash
fab build-web-mlp
fab run-server-web-mlp
```

In both cases open a browser and go to [127.0.0.1:8080](http://127.0.0.1:8080)
