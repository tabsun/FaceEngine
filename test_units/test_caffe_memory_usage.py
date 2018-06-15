import sys
import numpy as np
import cv2
sys.path.insert(0,'../backend_api')
from DLIB_detector import DLIB_detector
from RESNET_recognizor import RESNET_recognizor


if __name__=='__main__':
    model_path = '../backend_api/save_model'
    print "Begin loading..."
    detector = DLIB_detector(model_path, minsize=80)
    print "Load detector finish"
    featurer = RESNET_recognizor(model_path, gpu_id=0, dims=[512])
    print "Load resnet101 finish"

    image = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
    rects, shapes = detector.calc_landmark_result(image)
    shapes = detector.extract_five_result(shapes)

    extend_images = []
    for i in xrange(len(shapes)):
	extend_images.append(image)
    
    while(True):
	t = cv2.getTickCount()
	features = featurer.get_feature(np.array(extend_images), shapes)
	t = cv2.getTickCount() - t
	assert(features.shape[0] == shapes.shape[0])
	assert(features.shape[1] == 512)
	print "{}ms ".format(t*1000.0/cv2.getTickFrequency())
