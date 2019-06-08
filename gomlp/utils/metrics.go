package utils

import "math"

// GetAccuracy return the accuracy of a prediction (TP + TN) / (P + N)
func GetAccuracy(predictions [][]float64, expected [][]float64) float64 {
	tp := 0
	tn := 0

	for i := 0; i < len(predictions); i++ {
		for j := 0; j < len(predictions[0]); j++ {
			if predictions[i][j] == 0.0 && expected[i][j] == 0.0 {
				tn++
			} else if predictions[i][j] == 1.0 && expected[i][j] == 1.0 {
				tp++
			}
		}
	}

	return float64((tp + tn)) / math.Pow(float64(len(predictions)), 2)
}
