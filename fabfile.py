from fabric import task

@task
def build_web_mlp(c):
    """
        Build website that runs a MLP. Uoutput will be in mlp/web/dist
    """    
    with c.cd("mlp/web"):
        print("Building website")
        c.run("npm run build", replace_env=False)

        print("Moving datasets to website")
        c.run("cp -r ../../datasets dist/")
    
    build_web_mlp_wasm(c)

@task
def build_web_mlp_wasm(c):
    """
        Build website that runs a MLP. Uoutput will be in mlp/web/dist
    """    
    with c.cd("mlp/web"):
        with c.cd("src/components/go"):
            print("Building wasm file")
            c.run("GOOS=js GOARCH=wasm go build -o wasm/main.wasm", replace_env=False)
        
        print("Moving wasm file to website")
        c.run("cp src/components/go/wasm/main.wasm dist")

@task
def run_server_web_mlp(c):
    """
        Starts serving webpage in mlp/web/dist
    """  
    with c.cd("mlp/web/dist"):
        c.run("goexec 'http.ListenAndServe(\":8081\", http.FileServer(http.Dir(\".\")))'", replace_env=False)