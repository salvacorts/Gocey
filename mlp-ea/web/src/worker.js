importScripts("/wasm_exec.js")

if (!WebAssembly.instantiateStreaming) { // polyfill
    WebAssembly.instantiateStreaming = async (resp, importObject) => {
        const source = await (await resp).arrayBuffer();
        return await WebAssembly.instantiate(source, importObject);
    };
}

const go = new self.Go();

WebAssembly.instantiateStreaming(fetch("/go/wasm/main.wasm"), go.importObject).then(async (result) => {
let mod = result.module;
let inst = result.instance;

// Run
await go.run(result.instance);
});