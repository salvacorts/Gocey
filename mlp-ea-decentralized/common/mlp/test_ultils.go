package mlp

import (
	"strings"

	"github.com/salvacorts/go-perceptron-go/model/neural"
)

func equalString(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if strings.Compare(v, b[i]) != 0 {
			return false
		}
	}
	return true
}

func equalFloat64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func equalPattern(a, b []neural.Pattern) bool {
	if len(a) != len(b) {
		return false
	}

	for i := 0; i < len(a); i++ {
		if a[i].SingleExpectation != b[i].SingleExpectation {
			return false
		}

		if a[i].SingleRawExpectation != b[i].SingleRawExpectation {
			return false
		}

		if !equalFloat64(a[i].MultipleExpectation, b[i].MultipleExpectation) {
			return false
		}

		if !equalFloat64(a[i].Features, b[i].Features) {
			return false
		}
	}

	return true
}
