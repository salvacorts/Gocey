package main

import (
	"syscall/js"
)

type WebLogger struct {
	Document js.Value
}

func makeWebLogger() WebLogger {
	return WebLogger{Document: js.Global().Get("document")}
}

func (cl WebLogger) Write(p []byte) (o int, err error) {
	logs := cl.Document.Call("getElementById", "logs")

	log := cl.Document.Call("createTextNode", string(p))
	new := cl.Document.Call("createElement", "P")

	new.Call("appendChild", log)
	logs.Call("appendChild", new)

	return 0, nil
}
