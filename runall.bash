#!/bin/bash

python3 train.py gin-gat drugcombdb --device=cpu
python3 train.py gin-gat drugcombdb --device=cpu --validation=True
python3 train.py gin-gat drugcomb --device=cpu --validation=True
python3 train.py gin-gat drugcomb --device=cpu

python3 train.py ultra drugcombdb --device=cpu
python3 train.py ultra drugcomb --device=cpu
python3 train.py ultra drugcomb --device=cpu --validation=True
python3 train.py ultra drugcombdb --device=cpu --validation=True



