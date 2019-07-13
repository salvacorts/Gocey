const logs = document.getElementById("logs")
var chartFitness_ctx = document.getElementById('chartFitness').getContext('2d');
var chartNeurons_ctx = document.getElementById('chartNeurons').getContext('2d');

const worker = new Worker("worker.js")

avg = []
fitness = []
neurons = []
generations = []

worker.onmessage = function(e) {
    json = JSON.parse(e.data)

    // Append log to log box
    text = json.msg + ". Neurons: " + json.HiddenLayer_Neurons
    p = document.createElement("P")
    p.appendChild(document.createTextNode(text))
    logs.appendChild(p)
    logs.scrollTop = logs.scrollHeight
    console.log(text)

    // Append data to charts
    avg.push(json.Avg)
    fitness.push(json.Fitness)
    generations.push(json.Generation)
    neurons.push(json.HiddenLayer_Neurons)
    chartFitness.update()
    chartNeurons.update()
}

const chartFitness = new Chart(chartFitness_ctx, {
    type: 'line',
    data: {
        labels: generations,
        datasets: [{
            label: 'Best Fitness',
            data: fitness,
            backgroundColor: "#00b894",
            borderColor: "#00b894",
            fill: false,
            lineTension: 0,
        }, {
            label: 'Avg Fitness',
            data: avg,
            backgroundColor: "#8e5ea2",
            borderColor: "#8e5ea2",
            fill: false,
            lineTension: 0,
        }]
    },
    options: {
        responsive: false,
        title: {
            display: true,
            text: 'Fitness'
        },
        tooltips: {
            mode: 'index',
            intersect: true,
        },
        hover: {
            mode: 'nearest',
            intersect: true
        },
        scales: {
            xAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Generations'
                }
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Error'
                }
            }]
        }
    }
});

const chartNeurons = new Chart(chartNeurons_ctx, {
    type: 'line',
    data: {
        labels: generations,
        datasets: [{
            label: 'Best Neurons in hidden layer',
            data: neurons,
            backgroundColor: "#3e95cd",
            borderColor: "#3e95cd",
            fill: false,
            lineTension: 0,
        }]
    },
    options: {
        responsive: false,
        title: {
            display: true,
            text: 'Hidden Layer Neurons'
        },
        tooltips: {
            mode: 'index',
            intersect: true,
        },
        hover: {
            mode: 'nearest',
            intersect: true
        },
        scales: {
            xAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Generations'
                }
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Neurons'
                }
            }]
        }
    }
});
