from fabric import task

@task
def build_web_mlp(c):
    """
        Build website that runs a MLP. Uoutput will be in mlp/web/dist
    """    
    with c.cd("mlp/web"):
        print("[++] Building website")
        c.run("npm run build", replace_env=False)

        print("[++] Moving datasets to website")
        c.run("cp -r ../../datasets dist/")
    
    build_web_mlp_wasm(c)
    test_web(c)

@task
def build_web_mlp_wasm(c):
    """
        Build wasm file for Go. Output goes to mlp/web/dist
    """    
    test_common(c)

    with c.cd("mlp/web"):
        with c.cd("src/components/go"):
            print("[++] Building wasm file")
            c.run("GOOS=js GOARCH=wasm go build -o wasm/main.wasm", replace_env=False)
        
        print("[++] Moving wasm file to website")
        c.run("cp src/components/go/wasm/main.wasm dist")

@task
def run_server_web_mlp(c):
    """
        Starts serving webpage in mlp/web/dist
    """  
    with c.cd("mlp/web/dist"):
        c.run("goexec 'http.ListenAndServe(\":8081\", http.FileServer(http.Dir(\".\")))'", replace_env=False)

@task 
def test_common(c):
    """
        Test code that is shared between native and web implementation
    """
    with c.cd("mlp/common/tests"):
        print("[++] Testing common implementation")
        c.run("go test -v", replace_env=False)

@task
def test_web(c):
    """
        Test the website using SeleniumHQ chrome web-driver
    """
    with c.cd("mlp/web"):
        print("[++] Testing web with SeleniumHQ")
        c.run("selenium-side-runner tests/web.side", replace_env=False)

@task
def setup_raspbian(c):
    return