package utils

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

// OneHotEncode transforms the input feature into an array using one-hot-encoding
func OneHotEncode(feature []float64, n int) [][]float64 {
	output := make([][]float64, len(feature))

	for i, value := range feature {
		output[i] = make([]float64, n)

		for j := 0; j < n; j++ {
			if value != float64(j) {
				output[i][j] = 0
			} else {
				output[i][j] = 1
			}
		}
	}

	return output
}

// LoadCSV reads a csv dataset and returns a matrix of float64 with the information
func LoadCSV(filename string) [][]float64 {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Cannot open '%s' : error %s\n", filename, err.Error())
	}

	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ','

	rows, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Cannot read CSV data: %s", err.Error())
	}

	data := make([][]float64, len(rows))

	for i, row := range rows {
		data[i] = make([]float64, len(row))

		for j, item := range row {
			value, err := strconv.ParseFloat(item, 64)
			if err != nil {
				log.Fatalf("Cannot parse '%s' item to float64: %s", item, err.Error())
			}

			data[i][j] = value
		}
	}

	return data
}
