from fabric import task

@task(optional=['test'])
def build_web_mlp(c, test=False):
    """
        Build website that runs a MLP. Uoutput will be in mlp/web/dist
    """    
    with c.cd("web"):
        if c.run("test -d node_modules", warn=True).failed:
            print("[++] Installing node modules")
            c.run("npm install", replace_env=False)

        print("[++] Building website")
        c.run("npm run build", replace_env=False)

        print("[++] Moving datasets to website")
        c.run("cp -r ../../datasets dist/")
    
    build_web_mlp_wasm(c, test)

    if test:
        test_web(c)

@task(optional=['test'])
def build_web_mlp_wasm(c, test=False):
    """
        Build wasm file for Go. Output goes to mlp/web/dist
    """    
    if test:
        test_common(c)

    with c.cd("web/src/go"):
        print("[++] Building wasm file")
        c.run("GOOS=js GOARCH=wasm go build -o wasm/main.wasm", replace_env=False)

@task
def run_server_web_mlp(c):
    """
        Starts serving webpage in mlp/web/dist
    """  
    with c.cd("web/src"):
        c.run("goexec 'http.ListenAndServe(\":8081\", http.FileServer(http.Dir(\".\")))'", replace_env=False)

@task 
def test_common(c):
    """
        Test code that is shared between native and web implementation
    """
    with c.cd("common/tests"):
        print("[++] Testing common implementation")
        c.run("go test -v", replace_env=False)

@task
def test_web(c):
    """
        Test the website using SeleniumHQ chrome web-driver
    """
    with c.cd("web"):
        print("[++] Testing web with SeleniumHQ")
        c.run("selenium-side-runner tests/web.side", replace_env=False)

@task
def setup_raspbian(c):
    return