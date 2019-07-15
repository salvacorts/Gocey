package main

import (
	"syscall/js"
)

type WebLogger struct{}

func (cl WebLogger) Write(p []byte) (o int, err error) {
	js.Global().Call("postMessage", string(p))

	return 0, nil
}
