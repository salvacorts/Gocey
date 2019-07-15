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

    round = function(num) {
        return Math.floor(num * 100) / 100
    }

    // Append log to log box
    text = json.msg + ". Neurons: " + json.HiddenLayer_Neurons
    logs.innerHTML += "<br/>" + text
    logs.scrollTop = logs.scrollHeight
    console.log(text)

    // Append data to charts
    avg.push(round(json.Avg))
    fitness.push(round(json.Fitness))
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
            label: 'Best Solution',
            data: fitness,
            backgroundColor: "#00b894",
            borderColor: "#00b894",
            fill: false,
            lineTension: 0,

            datalabels: {
                color: '#00b894',
                align: 'top',
            }
        }, {
            label: 'Avg',
            data: avg,
            backgroundColor: "#8e5ea2",
            borderColor: "#8e5ea2",
            fill: false,
            lineTension: 0,

            datalabels: {
                color: '#8e5ea2',
                align: 'top',
            }
        }]
    },
    options: {
        responsive: true,
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
            label: 'Best solution',
            data: neurons,
            backgroundColor: "#3e95cd",
            borderColor: "#3e95cd",
            fill: false,
            lineTension: 0,

            datalabels: {
                color: '#3e95cd',
                align: 'top',
            }
        }]
    },
    options: {
        responsive: true,
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
        },
    }
});
