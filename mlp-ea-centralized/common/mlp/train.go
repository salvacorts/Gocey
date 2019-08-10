package mlp

import (
	mn "github.com/salvacorts/go-perceptron-go/model/neural"
)

// TransferF is the tranfer function
type TransferF = func(float64) float64

// Execute a multi layer Perceptron neural network.
// [mlp:MultiLayerNetwork] multilayer perceptron network pointer, [s:Pattern] input value
// It returns output values by network
func Execute(mlp *MultiLayerNetwork, s *mn.Pattern, tFunc TransferF, options ...int) (r []float64) {

	// new value
	nv := 0.0

	// result of execution for each OUTPUT NeuronUnit in OUTPUT NeuralLayer
	r = make([]float64, mlp.NeuralLayers[len(mlp.NeuralLayers)-1].Length)

	// show pattern to network =>
	for i := 0; i < len(s.Features); i++ {

		// setup value of each neurons in first layers to respective features of pattern
		mlp.NeuralLayers[0].NeuronUnits[i].Value = s.Features[i]
	}

	// init context
	for i := len(s.Features); i < int(mlp.NeuralLayers[0].Length); i++ {

		// setup value of each neurons in context layers to 0.5
		mlp.NeuralLayers[0].NeuronUnits[i].Value = 0.5
	}

	// execute - hiddens + output
	// for each layers from first hidden to output
	for k := 1; k < len(mlp.NeuralLayers); k++ {

		// for each neurons in focused level
		for i := 0; i < int(mlp.NeuralLayers[k].Length); i++ {

			// init new value
			nv = 0.0

			// for each neurons in previous level (for k = 1, INPUT)
			for j := 0; j < int(mlp.NeuralLayers[k-1].Length); j++ {

				// sum output value of previous neurons multiplied by weight between previous and focused neuron
				nv += mlp.NeuralLayers[k].NeuronUnits[i].Weights[j] * mlp.NeuralLayers[k-1].NeuronUnits[j].Value

				/*log.WithFields(log.Fields{
					"level":                 "debug",
					"msg":                   "multilayer perceptron execution",
					"len(mlp.NeuralLayers)": len(mlp.NeuralLayers),
					"layer:  ":              k,
					"neuron: ":              i,
					"previous neuron: ":     j,
				}).Debug("Compute output propagation.")
				/**/

			}

			// add neuron bias
			nv += mlp.NeuralLayers[k].NeuronUnits[i].Bias

			// compute activation function to new output value
			mlp.NeuralLayers[k].NeuronUnits[i].Value = tFunc(nv)

			// save output of hidden layer to context if nextwork is RECURRENT
			if k == 1 && len(options) > 0 && options[0] == 1 {

				for z := len(s.Features); z < int(mlp.NeuralLayers[0].Length); z++ {

					/*log.WithFields(log.Fields{
						"level":                               "debug",
						"len z":                               z,
						"s.Features":                          s.Features,
						"len(s.Features)":                     len(s.Features),
						"len mlp.NeuralLayers[0].NeuronUnits": len(mlp.NeuralLayers[0].NeuronUnits),
						"len mlp.NeuralLayers[k].NeuronUnits": len(mlp.NeuralLayers[k].NeuronUnits),
					}).Debug("Save output of hidden layer to context.")
					*/

					mlp.NeuralLayers[0].NeuronUnits[z].Value = mlp.NeuralLayers[k].NeuronUnits[z-len(s.Features)].Value
				}

			}

			/*log.WithFields(log.Fields{
				"level":                 "debug",
				"msg":                   "setup new neuron output value after transfer function application",
				"len(mlp.NeuralLayers)": len(mlp.NeuralLayers),
				"layer:  ":              k,
				"neuron: ":              i,
				"outputvalue":           mlp.NeuralLayers[k].NeuronUnits[i].Value,
			}).Debug("Setup new neuron output value after transfer function application.")
			*/
		}
	}

	// get ouput values
	for i := 0; i < int(mlp.NeuralLayers[len(mlp.NeuralLayers)-1].Length); i++ {

		// simply accumulate values of all neurons in last level
		r[i] = mlp.NeuralLayers[len(mlp.NeuralLayers)-1].NeuronUnits[i].Value
	}

	return r

}

// BackPropagate algorithm for assisted learning. Convergence is not guaranteed and very slow.
// Use as a stop criterion the average between previous and current errors and a maximum number of iterations.
// [mlp:MultiLayerNetwork] input value		[s:Pattern] input value (scaled between 0 and 1)
// [o:[]float64] expected output value (scaled between 0 and 1)
// return [r:float64] delta error between generated output and expected output
func BackPropagate(mlp *MultiLayerNetwork, s *mn.Pattern, o []float64, tFunc TransferF, tFuncD TransferF, options ...int) (r float64) {

	// var no []float64

	// init error
	e := 0.0

	// backpropagate error to previous layers
	// for each layers starting from the last hidden (len(mlp.NeuralLayers)-2)
	for k := len(mlp.NeuralLayers) - 2; k >= 0; k-- {

		// compute actual layer errors and re-compute delta
		for i := 0; i < int(mlp.NeuralLayers[k].Length); i++ {

			// reset error accumulator
			e = 0.0

			// for each link to next layer
			for j := 0; j < int(mlp.NeuralLayers[k+1].Length); j++ {

				// sum delta value of next neurons multiplied by weight between focused neuron and all neurons in next level
				e += mlp.NeuralLayers[k+1].NeuronUnits[j].Delta * mlp.NeuralLayers[k+1].NeuronUnits[j].Weights[i]

			}

			// compute delta for each neuron in focused layer as error * derivative of transfer function
			mlp.NeuralLayers[k].NeuronUnits[i].Delta = e * tFuncD(mlp.NeuralLayers[k].NeuronUnits[i].Value)

		}

		// compute weights in the next layer
		// for each link to next layer
		for i := 0; i < int(mlp.NeuralLayers[k+1].Length); i++ {

			// for each neurons in actual level (for k = 0, INPUT)
			for j := 0; j < int(mlp.NeuralLayers[k].Length); j++ {

				// sum learning rate * next level next neuron Delta * actual level actual neuron output value
				mlp.NeuralLayers[k+1].NeuronUnits[i].Weights[j] +=
					mlp.LRate * mlp.NeuralLayers[k+1].NeuronUnits[i].Delta * mlp.NeuralLayers[k].NeuronUnits[j].Value

			}

			// learning rate * next level next neuron Delta * actual level actual neuron output value
			mlp.NeuralLayers[k+1].NeuronUnits[i].Bias += mlp.LRate * mlp.NeuralLayers[k+1].NeuronUnits[i].Delta

		}

		// copy hidden output to context
		if k == 1 && len(options) > 0 && options[0] == 1 {

			for z := len(s.Features); z < int(mlp.NeuralLayers[0].Length); z++ {

				// save output of hidden layer to context
				mlp.NeuralLayers[0].NeuronUnits[z].Value = mlp.NeuralLayers[k].NeuronUnits[z-len(s.Features)].Value

			}

		}

	}

	// // compute global errors as sum of abs difference between output execution for each neuron in output layer
	// // and desired value in each neuron in output layer
	// for i := 0; i < len(o); i++ {

	// 	r += math.Abs(no[i] - o[i])

	// }

	// // average error
	// r = r / float64(len(o))

	return
}

// Training a mlp MultiLayerNetwork with BackPropagation algorithm for assisted learning.
func Training(mlp *MultiLayerNetwork, patterns []mn.Pattern, mapped []string, epochs int, bp ...bool) {

	epoch := 0
	output := make([]float64, len(mapped))

	var (
		tFunc  TransferF
		tFuncD TransferF
	)

	switch mlp.TFunc {
	case TransferFunc_SIGMOIDAL:
		tFunc = mn.SigmoidalTransfer
		tFuncD = mn.SigmoidalTransferDerivate
		// TODO: Add others here
	}

	// for fixed number of epochs
	for {

		// for each pattern in training set
		for _, pattern := range patterns {

			// setup desired output for each unit
			for io := range output {
				output[io] = 0.0
			}

			// setup desired output for specific class of pattern focused
			output[int(pattern.SingleExpectation)] = 1.0

			// Execute network with pattern passed over each level to output
			no := Execute(mlp, &pattern, tFunc)

			// init error
			e := 0.0

			// compute output error and delta in output layer
			for i := 0; i < int(mlp.NeuralLayers[len(mlp.NeuralLayers)-1].Length); i++ {

				// compute error in output: output for given pattern - output computed by network
				e = output[i] - no[i]

				// compute delta for each neuron in output layer as:
				// error in output * derivative of transfer function of network output
				mlp.NeuralLayers[len(mlp.NeuralLayers)-1].NeuronUnits[i].Delta = e * tFuncD(no[i])
			}

			// back propagation
			if len(bp) > 0 && bp[0] {
				BackPropagate(mlp, &pattern, output, tFunc, tFuncD)
			}
		}

		/*log.WithFields(log.Fields{
			"level":  "info",
			"place":  "validation",
			"method": "MLPTrain",
			"epoch":  epoch,
		}).Debug("Training epoch completed.")
		/**/

		// if max number of epochs is reached
		if epoch > epochs {
			// exit
			break
		}
		// increase number of epoch
		epoch++

	}
}
