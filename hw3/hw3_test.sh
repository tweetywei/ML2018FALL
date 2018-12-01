#!/bin/bash
wget -O EnsembleModelfinal.h5 https://www.dropbox.com/s/3zccq574rs4bbq5/EnsembleModelfinal.h5?dl=1
python3 predict_on_deepq.py $1 $2