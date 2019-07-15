import React from 'react'

export default class App extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      isLoading: true
    }
  }
  componentDidMount() {
    if (!WebAssembly.instantiateStreaming) { // polyfill
			WebAssembly.instantiateStreaming = async (resp, importObject) => {
				const source = await (await resp).arrayBuffer();
				return await WebAssembly.instantiate(source, importObject);
			};
    }
    
    const go = new Go();
    WebAssembly.instantiateStreaming(fetch("main.wasm"), go.importObject).then(async (result) => {
      let mod = result.module;
      let inst = result.instance;

      // Run
      this.setState({ isLoading: false })
      await go.run(result.instance);
    });
  }
  render() {
    return this.state.isLoading 
            ? <div>Loading</div> 
            :  <div id="logs"></div>
  }
}