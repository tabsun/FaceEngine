# here is the comments for config.json
# but json cannot include comments in its file, so I wrote the comments into this ReadMe

###########################################
# scorer_params 
#  - type : ["JOINTBAY" or Not set]
#  - dims : [128 or 512 or 5900]
#           when type is not set, dims means possible feature length - 128 for facenet and 512 for resnet101
#                                                                      5900 for Jointbay PCA input feature dimension
# feature_params
#  - type :     ["resnet101" or "facenet"] face recognition model type
#  - gpu_id :   [int] which gpu to be used
#  - dims :     [{"facenet":[128, 768],"resnet101":[512]}] allowed feature dimensions to extract
#  - batchsize: [int] max input image batch
# detector_params
#  - type :    ["MTCNN" or "DLIB"] face detection model type 
#  - minSize : [int] min detected face size
#  - factor :  [0.5] only needed when type = MTCNN 
#  - threshold:[[0.8, 0.8, 0.9]] only needed when type = MTCNN
#  - gpu_id:   [int] only needed when type = MTCNN
# attricalcor_params
#  - type :   ["tensorflow_attr" or "caffe_attr"] face attribute detection model type
#  - gpu_id : [int] gpu to be used
# anglejudger_params
#  - type : ["dynamic" or "fix"] actually we recommend always use dynamic because of accurity
# ajuster_params
#  - type : ["tensorflow_106" or "caffe_106"] 106-face landmarks detection model type
############################################
{
  "scorer_params":
        {"type": "Jointbay", "dims": [128, 512]},
  "featurer_params":
        {"type": "resnet101", "gpu_id": 0, "dims": {"facenet":[128, 768],"resnet101":[512]}, "batchsize": 20},
  "detector_params":
        {"type": "DLIB", "minSize": 160, "factor": 0.5, "threshold":[0.8, 0.8, 0.9], "gpu_id": 0},
  "attricalcor_params":
        {"type": "caffe_attr", "gpu_id": 0},
  "anglejudger_params":
        {"type": "dynamic"},
  "ajuster_params":
        {"type": "tensorflow_106"}
}

# NOTES:
# [1] If you want to detect full result including pose data, you need to set detector to DLIB. MTCNN cannot do this!
# [2] factor/threshold/gpu_id in detector_params are required only when use MTCNN, if you use DLIB they can be removed!
# [3] For some reason now Jointbay is not used in scorer, it's in TODO list for future
