import sys, json, dlib, cv2, os
import numpy as np
from ANGLE_judger import ANGLE_judger
from ATTRI_judger import ATTRI_judger
#from JOINTBAY_judger import JOINTBAY_judger
from LANDMARK_ajuster import LANDMARK_ajuster

class FACE_engine:
    def __init__(self, model_path, config_file=None):
        if(config_file is None):
            self.config_params = { 'detector_params': \
			                             {'type':'DLIB', \
						     'minSize': 80}, \
				   'anglejudger_params':{'type':'dynamic'}, \
				   'attricalcor_params':{'type':'nn', \
				                        'gpu_id':0 }, \
				   'featurer_params':{'type':'both', \
						      'gpi_id': 0, \
				                      'batchsize': 1}, \
				   'scorer_params': \
				                     {'type':'JOINTBAY', \
						     'dim': 5900} \
				   }
        else:
	    if(os.path.exists(config_file)):
	        with open(config_file, 'r') as f:
		    self.config_params = json.load(f)
	    else:
	        raise Exception( "Config file {} do not exist.".format(config_file))
        
        det_params = self.config_params['detector_params']
	ang_params = self.config_params['anglejudger_params']
        att_params = self.config_params['attricalcor_params']
	fea_params = self.config_params['featurer_params']
	sco_params = self.config_params['scorer_params']
        aju_params = self.config_params['ajuster_params']
	########## 1st part - detector
	if(det_params['type'] == 'DLIB'):
            from DLIB_detector import DLIB_detector
            self.detector = DLIB_detector(model_path, det_params['minSize'])
	else:
            from MTCNN_detector import MTCNN_detector
            self.detector = MTCNN_detector(model_path, gpu_id=det_params['gpu_id'], \
                                           minsize=det_params['minSize'], factor=det_params['factor'], \
                                           threshold=det_params['threshold'])
	########## 2nd part - anglejudger
	self.anglejudger = ANGLE_judger(model_path, ang_params['type'])
	########## 3rd part - scorer
	#if(sco_params['type'] == 'JOINTBAY'):
	#    self.scorer = JOINTBAY_judger(model_path, dim=sco_params['dim'])
        ########## 4th part - featurer
        if(fea_params['type'] == 'resnet101'):
	    from RESNET_recognizor import RESNET_recognizor
	    self.featurer = RESNET_recognizor(model_path, gpu_id=fea_params['gpu_id'], dims=fea_params['dims']['resnet101'])
            image = cv2.imread(os.path.join(model_path,"test.jpg"), cv2.IMREAD_COLOR)
            with open(os.path.join(model_path, "first_usage_shape.txt")) as f:
                shapes = np.array([int(s) for s in f.readline().split(" ")[0:-1]]).reshape(1,10)
            self.featurer.get_feature(np.array([image]), shapes)
        if(fea_params['type'] == 'facenet' or att_params['type'] == "tensorflow_attr"):
            from FACENET_recognizor import FACENET_recognizor 
	    self.featurer2 = FACENET_recognizor(model_path, gpu_id=fea_params['gpu_id'], \
                                                dims=fea_params['dims']['facenet'], batchsize=fea_params['batchsize'])
        
	########## 5th part - attrijudger
	self.attrijudger = ATTRI_judger(model_path, gpu_id=att_params['gpu_id'], type_str=att_params['type'])
	########## 6th part - ajuster
	self.ajuster = LANDMARK_ajuster(model_path, aju_params['type'])

    def destroy(self):
        if self.detector and self.config_params['detector_params']['type'] == 'MTCNN':
	    self.detector.destroy()
	if self.featurer2:
	    self.featurer2.destroy()
	if self.attrijudger:
	    self.attrijudger.destroy()

    def calc_det_result(self, image):
        return self.detector.calc_det_result(image)

    def calc_landmark_result(self, image):
        rects, shapes = self.detector.calc_landmark_result(image)
        return rects, shapes

    # SAME as calc_landmark_resut but add the pose results
    # Return :  a numpy array of shape [num_face, 3]
    # Denote of each row:
    #           [pose_x, pose_y, pose_z] all measured in degrees and image's left_top_clockwise is positive pole
    def calc_full_result(self, image):
        if(self.config_params['detector_params']['type'] != 'DLIB'):
	    print "If you want the full results including the pose data you must set detector type is DLIB in config file."
	    return None
        else:
	    rects, shapes = self.detector.calc_landmark_result(image)
	    poses = []
	    height, width = image.shape[0:2]
	    anglejudger_type = self.config_params['anglejudger_params']['type']
	    for shape in shapes:
	    	pose = self.anglejudger.CalcPose(shape, width, height, type=anglejudger_type)
	    	pose = ["%.0f" % elem for elem in pose]
	    	poses.append(pose)
	    poses = np.array(poses)
	    return rects, shapes, poses
	        
    # image - the whole image
    # shape - n x 10 in which each row denotes left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, ..._y, left_mouthcorner_x,...y, right_..._x, ..._y
    # return - n x 212 in which each row denotes x0,y0,x1,y1...
    def calc_106_landmarks(self, image, shapes):
        shapes = self.detector.extract_five_result(shapes)
	shapes = self.ajuster.calc_106_landmarks(image, shapes)
	return shapes

    # ATTRI_judger have two mode: caffe and tensorflow version
    # The caffe version will extract attribute datas from image 
    # while tensorflow version will extract these from face_id feature, which is extracted with layer_id=[0,1]
    def calc_attribute_result(self, image, shapes):
        results = []
        if(self.attrijudger.type_str == "tensorflow_attr"):
            features = self.calc_feature_result(image, shapes, layer_ids=[0,1])
            for feature in features:
	        result = self.attrijudger.get_result(feature=feature)
                results.append(result)
        else:
            shapes = self.detector.extract_five_result(shapes)
            for shape in shapes:
                result = self.attrijudger.get_result(image=image, shape=shape)
                results.append(result)
	return np.array(results)

    # images - can be a (h,w,c) image or (b,h,w,c) image batch
    # shapes - numpy array of detected shape from detector, the position is responding to images
    # type  - "facenet" will return facenet features
    #         while "resnet101" will return resnet101 features
    # layer_ids : 0 - output 128-dim feature
    #             1 - baoming's 640-dim feature
    #             if you want both you should set layer_ids = [0,1]
    #             if you want [baoming's feature, 128-dim output], then you should set layer_ids = [1, 0]
    # features - numpy array of (num, dim)
    def calc_feature_result(self, images, shapes, layer_ids=[0]):
	# Prepare all the data formats into a list of images and a numpy array with the same len()
        if(len(shapes) == 0 or np.prod(images.shape) == 0):
            return np.array([])
        extend_images = []
	extend_images = []
        if(type(images) is np.ndarray):
            if( len(images.shape) == 3 ):
                for i in xrange(len(shapes)):
                    extend_images.append(images)
            if( len(images.shape) == 4 and images.shape[0] == 1 and shapes.shape[0] != 1):
                for i in xrange(shapes.shape[0]):
                    extend_images.append(images[0])
            if( len(images.shape) == 4 and images.shape[0] == shapes.shape[0]):
                for i in xrange(shapes.shape[0]):
                    extend_images.append(images[i,...])
        else:
            if( len(images) == shapes.shape[0] ):
                for i in xrange(shapes.shape[0]):
                    extend_images.append(images[i])
	assert( len(extend_images) == len(shapes) )

	if( (self.config_params['featurer_params']['type'] == "facenet" and set(layer_ids)==set([0]) ) or \
            (self.config_params['attricalcor_params']['type'] == "tensorflow_attr" and set(layer_ids)==set([0,1]) ) ):
            shapes = self.detector.extract_eye_result(shapes)
	    features = self.featurer2.get_feature(np.array(extend_images), shapes, layer_ids)
            return features
        if(self.config_params['featurer_params']['type'] == "resnet101"):
	    shapes = self.detector.extract_five_result(shapes)
	    features = self.featurer.get_feature(np.array(extend_images), shapes)
	    return features

    # Calc the score between two features
    # Larger score means more similar
    def calc_score_result(self, feat1, feat2):
        type_str=self.config_params['featurer_params']['type']
	dims = self.config_params['scorer_params']['dims']
	assert(np.prod(feat1.shape) == np.prod(feat2.shape))
	assert(np.prod(feat1.shape) in dims)
	if(type_str == 'Jointbay'):
            print "Please do not use Jointbay for now"
            return 0.0
	    #return self.scorer.get_result(feat1, feat2)	
	else:
	    if(type_str == "resnet101"):
                return self.featurer.get_score(feat1, feat2)
	    if(type_str == "facenet"):
		return self.featurer2.get_score(feat1, feat2)
	    else:
		raise Exception("score type is incorrect, must be resnet101 facenet or Jointbay")
