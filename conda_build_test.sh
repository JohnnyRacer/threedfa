#!/bin/bash
conda create -y -n tdfatest python=3.8;
conda activate tdfatest;
pip install build;
python -m build;
pip install $(find dist -name '*cp38*');
which python;
python tests/basic_test.py;
conda activate;
conda env remove -y -n tdfatest;