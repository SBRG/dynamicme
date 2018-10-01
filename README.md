# dynamicme
Dynamic simulation of models of Metabolism and macromolecular Expression

## Installation
1. Run `python setup.py install`
1. Open `notebooks/publication/start_here.ipynb` to run your first dynamicME simulation.

## Requirements
DynamicME requires
- Python version >= 2.7.
  - *Note:* Python 3 compatibility: scheduled Q4 2018.
- COBRApy versions <= 0.5.11. We recommend using 0.5.11
- [COBRAme](https://github.com/SBRG/cobrame)
- [ECOLIme](https://github.com/SBRG/ecolime)
- [solvemepy](https://github.com/SBRG/solvemepy)

#### Note on version compatibility
We are working on bringing DynamicME up to date with the latest COBRApy.
Please note that DynamicME is built on COBRAme, and work is underway to first make
COBRAme compatible with the latest backend updates that were made in COBRApy after 0.5.11.

## Citing:
If you find the module useful, please consider citing:
- [Yang et al. (2018) bioRxiv. doi:10.1101/319962](https://www.biorxiv.org/content/early/2018/05/15/319962)
