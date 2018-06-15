#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/your/caffe/python
python server.py 
#nohup python server.py >log.txt &
