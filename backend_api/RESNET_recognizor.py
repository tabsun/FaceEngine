import os, caffe, cv2, nudged, scipy
import numpy as np

class RESNET_recognizor:
    # model_path  -  the path including ./  ResNet101-FaceNet.prototxt
    #                                       resnet101_baseline_fullface_final_102x102.caffemodel
    # gpu_id      -  the specific gpu which is used, if gpu_id < 0 then cpu is used instead of gpu
    # dims        -  the extracted feature's possible dimensions 
    def __init__(self, model_path, gpu_id=0, dims=[512]):
	assert(os.path.exists(os.path.join(model_path, 'ResNet101-FaceNet.prototxt')))
	assert(os.path.exists(os.path.join(model_path, 'resnet101_baseline_fullface_final_102x102.caffemodel')))
        self.align_size = (112, 112)
	self.input_size = (102, 102)
	self.dims = dims
	self.anchor_points = np.array([35, 32, 77, 32, 56, 59, 39, 79, 73, 79]).reshape(-1,2).astype(float)
        self.net = caffe.Net(os.path.join(model_path, 'ResNet101-FaceNet.prototxt'), \
			os.path.join(model_path, 'resnet101_baseline_fullface_final_102x102.caffemodel'), \
			caffe.TEST)
	self.net.blobs['data'].reshape(2, 3, self.input_size[0], self.input_size[1])
	if(gpu_id < 0):
	    caffe.set_mode_cpu()
	else:
            caffe.set_device(gpu_id)
	    caffe.set_mode_gpu()

    def _align_image(self, image, shape):
	shape = np.array(shape)
	assert(np.prod(shape.shape) == 10)
        shape = shape.reshape((5,2))
        tform = nudged.estimate(shape, self.anchor_points)
        tform = np.asarray(tform.get_matrix())
        img_align = cv2.warpPerspective(image, tform, self.align_size) 
        img_resize = cv2.resize(img_align, self.input_size)
        return img_resize
    
    # images  -  the images to extract feature, attention for one point: image is got by cv2.imread as it's BGR-channel sequence.
    # shapes  -  the detected faces' landmarks, a numpy array with size N x 10:
    #               [left_eye_x,left_eye_y,right_eye_x,right_eye_y, \
    #                nose_x,nose_y, left_mouth_x,left_mouth_y, right_mouth_x, right_mouth_y]
    # return  -  the corresponding feature stored in a numpy array with size (N, fea_dim)
    def get_feature(self, images, shapes):
	assert(len(images) == len(shapes))
	assert(shapes.shape[1] == 10)
	features = np.array([])
	for i in xrange(len(shapes)):
	    image = self._align_image(images[i,...], shapes[i,:])
            img_array = []
            img_array.append(image.transpose(2,0,1).astype(float)-128.0)
            img_flip = cv2.flip(image, 1)
            img_array.append(img_flip.transpose(2,0,1).astype(float)-128.0)
    
            self.net.blobs['data'].data[:,...] = np.asarray(img_array).astype(float)
            self.net.reshape()
            output = self.net.forward()
            featV = output['feat']
            featV = np.amax(featV.reshape(2, -1),axis=0).reshape(1,-1)
	    features = np.vstack([features, featV]) if features.size else featV
	assert(features.shape[1] in self.dims)
        return features

    def get_score(self, a, b):
        assert(np.prod(a.shape) == np.prod(b.shape))
	assert(np.prod(a.shape) in self.dims)
        s = np.dot(a.reshape((1,-1)),b.reshape((-1,1)))/(np.linalg.norm(a) * np.linalg.norm(b))
        if( s<0.5 ):
            s = (s + 1.0)/3.0
        return s
