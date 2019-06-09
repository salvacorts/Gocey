package main

import (
	"github.com/dennwc/dom"
)

type WebLogger struct{}

func (cl WebLogger) Write(p []byte) (o int, err error) {
	logs := dom.Doc.GetElementById("logs")

	new := dom.Doc.CreateElement("p")
	new.SetTextContent(string(p))
	logs.AppendChild(new)

	return 0, nil
}
