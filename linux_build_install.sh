#! /bin/sh
!which python;
pip install -r requirements.txt;
python -m build;
raw_ver=$(python -V | sed 's/[^0-9]*//g'); # Gets the raw version of python and filter it through sed
fil_ver=${raw_ver:0:2}; #Truncates the string 
pip install dist/threedfa-0.0.1-cp${fil_ver}-cp${fil_ver}-$(uname | awk '{print tolower($0)}')_$(arch).whl --force-reinstall;
