const logs = document.getElementById("logs")
var chartLocal_ctx = document.getElementById('chartLocal').getContext('2d');
var chartRemote_ctx = document.getElementById('chartRemote').getContext('2d');

const worker = new Worker("worker.js")

// Local stats
fitness = []
evaluations = []

// Remote stats
totalEvaluations = []
avgFitness = []
bestFitness = []

worker.onmessage = function(e) {
    json = JSON.parse(e.data)

    round = function(num) {
        return Math.floor(num * 100) / 100
    }

    // Append log to log box
    text = json.msg
    logs.innerHTML += "<br/>" + text
    logs.scrollTop = logs.scrollHeight
    console.log(text)

    // Append data to charts
    switch(json.Scope) {
        case "local":
            evaluations.push(json.Evaluations)
            fitness.push(round(json.Fitness))
            chartLocal.update()
            break;
        case "remote":
            totalEvaluations.push(json.Evaluations)
            avgFitness.push(round(json.AvgFitness))
            bestFitness.push(round(json.BestFitness))
            chartRemote.update()
          break;
      }
}

const chartLocal = new Chart(chartLocal_ctx, {
    type: 'line',
    data: {
        labels: evaluations,
        datasets: [{
            label: 'Fitness of evaluated individuals',
            data: fitness,
            backgroundColor: "#00b894",
            borderColor: "#00b894",
            fill: false,
            lineTension: 0,
        }]
    },
    options: {
        responsive: true,
        title: {
            display: true,
            text: 'Local Statistics'
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
                    labelString: 'Evaluations'
                }
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Fitness'
                }
            }]
        }
    }
});

const chartRemote = new Chart(chartRemote_ctx, {
    type: 'line',
    data: {
        labels: totalEvaluations,
        datasets: [{
            label: 'Best solution',
            data: bestFitness,
            backgroundColor: "#3e95cd",
            borderColor: "#3e95cd",
            fill: false,
            lineTension: 0,
        },
        {
            label: 'Average',
            data: avgFitness,
            backgroundColor: "#6c5ce7",
            borderColor: "#6c5ce7",
            fill: false,
            lineTension: 0,
        }]
    },
    options: {
        responsive: true,
        title: {
            display: true,
            text: 'Server Statistics'
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
                    labelString: 'Evaluations'
                }
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Fitness'
                }
            }]
        },
    }
});
