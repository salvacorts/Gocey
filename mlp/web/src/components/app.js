import React from 'react'

export default class App extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      isLoading: true
    }
  }
  componentDidMount() {
    const go = new Go();
   
    WebAssembly.instantiateStreaming(fetch("main.wasm"), go.importObject).then((result) => {
        go.run(result.instance);
        trainMLP()
        this.setState({ isLoading: false })
    });
  }
  render() {
    return this.state.isLoading 
            ? <div>Loading</div> 
            :  <div id="logs"></div>
  }
}