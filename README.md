# Face Engine 

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

_test image_ | _face detection_ |_landmark detection_|_age_|_gender_|_race_|_smile_|_celebrity_
:------------|:----------------:|:------------------:|:---:|:------:|:----:|:-----:|:----------:
<img width="150" src="https://wx1.sinaimg.cn/mw1024/89ef5361ly1fscaalub19j20j20j1dki.jpg">|<img width="150" src="https://wx1.sinaimg.cn/mw1024/89ef5361ly1fsc9lppvilj20j20j10xj.jpg">|<img width="150" src="https://wx3.sinaimg.cn/mw1024/89ef5361ly1fsc9lppimgj20j20j1af1.jpg">|24.47|female|asian|smile|Fan Bingbing
<img width="150" src="https://wx3.sinaimg.cn/mw1024/89ef5361ly1fsc9lph3uzj20re0re78i.jpg">|<img width="150" src="https://wx4.sinaimg.cn/mw1024/89ef5361ly1fsc9lpn00ij20re0re78p.jpg">|<img width="150" src="https://wx2.sinaimg.cn/mw1024/89ef5361ly1fscaahs4ylj20re0reaem.jpg">|60.15|male|asian|calm|Guo Degang
<img width="150" src="https://wx2.sinaimg.cn/mw1024/89ef5361ly1fsc9lozotnj20fu0fu77x.jpg">|<img width="150" src="https://wx1.sinaimg.cn/mw1024/89ef5361ly1fsc9lp41xcj20fu0fuwi6.jpg">|<img width="150" src="https://wx4.sinaimg.cn/mw1024/89ef5361ly1fsc9lpf35ej20fu0fu0wn.jpg">|26.06|female|asian|smile|XinYuanJieYi
<img width="150" src="https://wx4.sinaimg.cn/mw1024/89ef5361ly1fsc9m8n9q6j20fu0fugnf.jpg">|<img width="150" src="https://wx2.sinaimg.cn/mw1024/89ef5361ly1fsc9m8d72pj20fu0fugni.jpg">|<img width="150" src="https://wx3.sinaimg.cn/mw1024/89ef5361ly1fsc9m8hklbj20fu0fugno.jpg">|54.52|male|african|calm|AoBaMa

Wo De Nan Shen AAAAAAAAAAA!

Wo De Nv Shen  EEEEEEEEEEE!
