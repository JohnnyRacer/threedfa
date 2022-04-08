#! /bin/sh
pip install -r requirements.txt;
python -m build;
raw_ver=$(python -V | sed 's/[^0-9]*//g');
fil_ver=${raw_ver:0:2};
pip install dist/threedfa-0.0.1-cp${fil_ver}-cp${fil_ver}-$(uname | awk '{print tolower($0)}')_$(arch).whl --force-reinstall;
echo Compiled and installed threedfa
