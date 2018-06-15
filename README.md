# face_backend_svr

## Introduction

This project includes face detection / alignment / 5 or 68 or 106 landmarks detection / face attributes detection including age+gender+race+smile / face orientation detection / face recognition.

Basically it covers all fields of face tech except for face living detection which is also really important in the whole chain. But as it usually runs on mobile devices, so it's not contained in this repo which focused on backend usage.


## Requirements

tensorflow

caffe for python binding

numpy

opencv-python

dlib

scikit-learn

scikit-video

nudged

## Quick Start

I wrote a face server backend based on flask. Just run ./run.sh to start it and then you can visit the api provided.

More details can be found in ./backend_api/. 

ATTENSION for below settings:

CUDA_VISIBLE_DEVICES/PYTHONPATH/port setting, you may need modify this according to your env.

Any problem is welcome to report to issue or contact buptmsg@gmail.com.

## Overview

_face detection_ |_landmark detection_|_age_|_gender_|_race_|_smile_|_orientation_|_celebrity_
:---------------:|:------------------:|:---:|:------:|:----:|:-----:|:-----------:|:----------:

