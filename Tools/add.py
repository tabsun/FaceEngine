import os, cv2, base64, re
import tensorflow as tf
import numpy as np
from matio import load_mat, save_mat
from shutil import copyfile

# FACE ENGINE
import sys
sys.path.insert(0, '../backend_api')
from FACE_engine import FACE_engine

CELEIMAGE_FOLDER = './add'
CELEBACKUP_FOLDER = '../cele_images'
CELEPOOL_FOLDER = '../celefeature_files'
CELEHEAD_FOLDER = '../static/cele_images'
ALLOWED_EXTENSIONS = set([ 'png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG' ])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def get_name(filename):
    return filename.rsplit('#', 1)[0]

def log_info(text):
    with open("log.txt", "w") as f:
	f.write(text)
    return

def process():
    di = 0
    for the_file in os.listdir(CELEIMAGE_FOLDER):
	print the_file
	if(allowed_file(the_file)):
	    image = cv2.imread(os.path.join(CELEIMAGE_FOLDER, the_file), cv2.IMREAD_COLOR)
            rects, shapes, poses = eng.calc_full_result(image)

	    if(len(rects) == 0):
		continue
	    else:
	        features = eng.calc_feature_result(image, shapes, "resnet101")
		if(len(rects) != 1):
		    rect = [0,0,image.shape[1]-1,image.shape[0]-1]
		else:
		    rect = rects[0]
		    x,y,right,bottom,s = rect[:]
		    w = right - x
		    h = bottom -y
		    x = max(0, int(x-w*0.2))
		    y = max(0, int(y-h*0.2))
		    right = min(image.shape[1], int(right+w*0.2))
		    bottom = min(image.shape[0], int(bottom+h*0.2))
		    image = image[y:bottom, x:right, :]
		name = the_file.rsplit('#',1)[0]
		cv2.imwrite(os.path.join(CELEHEAD_FOLDER, name+'.jpg'), image)
                #copyfile(os.path.join(CELEIMAGE_FOLDER, the_file), os.path.join(CELEBACKUP_FOLDER, the_file))

		for i in xrange(features.shape[0]):
		    feature = features[i,:]
                    save_path = os.path.join(CELEPOOL_FOLDER, "{}.{}.bin".format(the_file,i))
	            save_mat(save_path, feature)
    return

if __name__=='__main__':
    # Params
    model_path = "../../../save_model"
    config_path = "../../../save_model/config.json"

    eng = FACE_engine(model_path, config_path)
    process()
    eng.destroy()
