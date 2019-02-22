## Build the Docker Container
```bash
docker build -t cs234-project .
```

## Run the Docker Container and Get a Shell
```bash
docker run -v $PWD:/project -p 8888:8888 -it cs234-project /bin/bash
```

## Start a Jupyter Notebook
```bash
jupyter notebook --allow-root --ip=0.0.0.0
```
