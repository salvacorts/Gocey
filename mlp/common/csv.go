package common

import (
	// sys import
	"encoding/csv"
	"io"
	"log"
	"strings"

	// this repo internal import
	. "github.com/made2591/go-perceptron-go/model/neural"
	mu "github.com/made2591/go-perceptron-go/util"
)

func LoadPatternsFromCSVFile(content string) ([]Pattern, error, []string) {

	// init patterns
	var patterns []Pattern

	// create pointer to read file
	pointer := csv.NewReader(strings.NewReader(content))

	var lineCounter int = 0
	// for each record in file
	for {

		// read line, check error
		line, err := pointer.Read()

		// if end of file reached, exit loop
		if err == io.EOF {
			break
		}

		// if another error encountered, exit program
		if err != nil {
			log.Fatalf("Failed to parse line. %s", err.Error())
			return patterns, err, nil
		}

		// line values cast to float64
		var floatingValues = mu.StringToFloat(line, 1, -1.0)

		// add casted pattern to training set
		patterns = append(
			patterns,
			Pattern{Features: floatingValues, SingleRawExpectation: line[len(line)-1]})

		lineCounter = lineCounter + 1

	}

	// cast expected values to float64 numeric values
	mapped := RawExpectedConversion(patterns)

	// return patterns
	return patterns, nil, mapped
}
