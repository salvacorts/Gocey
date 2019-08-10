package main

import "syscall/js"

// WebLogger write logs into the webpage DOM
type WebLogger struct{}

func (cl WebLogger) Write(p []byte) (o int, err error) {
	js.Global().Call("postMessage", string(p))

	return 0, nil
}
