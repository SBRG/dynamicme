# dynamicme
Dynamic simulation of models of Metabolism and macromolecular Expression

## Quickstart using Docker
We recommend using Docker, since manual installation may prove to be somewhat tedious.
1. Install Docker ([https://docs.docker.com/install/](https://docs.docker.com/install/))
1. In the command line, run `docker run -p 8888:8888 --rm -i -v $(pwd):/mount_point/ -t sbrg/dynamicme:master bash`.
This command will initiate a Docker container into the `/home/meuser` directory and mount the contents of the directory where the command was ran into the docker container at `/mount_point/`.
1. Start a jupyter notebook by running `jupyter noteook --ip=0.0.0.0 --port=8888`. Point the browser to `localhost:8888` and input the provided token to access the notebook.
1. Open `notebooks/publication/start_here.ipynb` to run your first dynamicME simulation.

## Installation
1. Run `python setup.py install`
1. Open `notebooks/publication/start_here.ipynb` to run your first dynamicME simulation.

## Requirements
DynamicME requires
- Python. Tested only 2.7.
- COBRApy versions <= 0.5.11. We recommend using 0.5.11
- [COBRAme](https://github.com/SBRG/cobrame)
- [ECOLIme](https://github.com/SBRG/ecolime)
- [solvemepy](https://github.com/SBRG/solvemepy)

## Citing:
If you find the module useful, please consider citing:
- [Yang et al. (2018) doi:10.1101/319962](https://www.biorxiv.org/content/early/2018/05/15/319962)
