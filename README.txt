# face_backend_svr

=============================
## Requirements

tensorflow

caffe for python binding

numpy, opencv-python, dlib, scikit-learn, scikit-video, nudged

## Quick Start

Just run ./run.sh

ATTENSION for below settings:

CUDA_VISIBLE_DEVICES/PYTHONPATH/port setting, you may need modify this according to your env

Some machine like aliyun's machine will be running at wrong time clock, notice this time_error in server.py

Any problem is welcome to report to issue or contact sunhaiyong@yijia.ai

## Overview

Face service backend api

backend_api  			-  	The FACE_engine interfaces

black_celefeature_files		- 	The politicians' face feature file on black list

celefeature_files 		- 	The celebrities' face feature file

server.py 			- 	Flask API definition

Tools 				- 	Tools to add one person into politician/celebrity pool

					or generate new face feature by new model provided by backend_api
