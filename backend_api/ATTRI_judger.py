import numpy as np
import os, nudged, cv2

class ATTRI_judger:
    def __init__(self, model_path, gpu_id=0, type_str="caffe_attr"):
        self.type_str = type_str
        if(type_str == "caffe_attr"):
            global caffe
            import caffe
            assert(os.path.exists(os.path.join(model_path, 'face_attr.deploy')))
            assert(os.path.exists(os.path.join(model_path, 'face_attr.caffemodel')))
            self.align_size = (112, 112)
            self.input_size = (120, 120)
            self.anchor_points = np.array([35, 32, 77, 32, 56, 59, 39, 79, 73, 79]).reshape(-1,2).astype(float)
            self.net = caffe.Net(os.path.join(model_path, 'face_attr.deploy'), \
                            os.path.join(model_path, 'face_attr.caffemodel'), \
                            caffe.TEST)
            self.net.blobs['data'].reshape(1, 3, self.input_size[0], self.input_size[1])
            if(gpu_id < 0):
                caffe.set_mode_cpu()
            else:
                caffe.set_device(gpu_id)
                caffe.set_mode_gpu()
        else:
            global tf
            import tensorflow as tf
	    if(not os.path.exists(os.path.join(model_path, 'face_attribute.meta'))):
                raise Exception("{} do not exists.".format(os.path.join(model_path, 'face_attribute.meta')))
            
            self.gpu_id = gpu_id
	    self.dim = 768
	    with tf.device('/gpu:{}'.format(gpu_id)):
                with tf.Graph().as_default():
	    	    self.sess = tf.Session()
	    	    saver = tf.train.import_meta_graph(os.path.join(model_path,'face_attribute.meta'))
                    self.sess.run(tf.global_variables_initializer())
	    	    saver.restore(self.sess, os.path.join(model_path,'face_attribute'))

    def _align_image(self, image, shape):
        shape = np.array(shape)
        assert(np.prod(shape.shape) == 10)
        shape = shape.reshape((5,2))
        tform = nudged.estimate(shape, self.anchor_points)
        tform = np.asarray(tform.get_matrix())
        img_align = cv2.warpPerspective(image, tform, self.align_size)
        img_resize = cv2.resize(img_align, self.input_size)
        return img_resize
	
    def get_result(self, image=None, shape=None, feature=None):
        if(self.type_str == "caffe_attr"):
            assert(image is not None and shape is not None)
            align_image = self._align_image(image, shape)
            align_image = align_image.transpose(2,0,1).astype(float)-128.0
            self.net.blobs['data'].data[:,...] = np.asarray([align_image]).astype(float)
            self.net.reshape()
            output = self.net.forward()
            
            attri_vec = output['prob'].flatten()
            result = []
            result.append(np.argmax(attri_vec[0:4])) 
            result.append(np.argmax(attri_vec[4:6]))
            result.append(np.argmax(attri_vec[6:9]))
            result.append(attri_vec[9])
        else:
	    assert(feature is not None and np.prod(feature.shape) == self.dim)
	    feature =feature.reshape((-1, self.dim))
            result = []
            with tf.device('gpu:{}'.format(self.gpu_id)):
	        score = self.sess.run(('softmax_race/add:0', 'softmax_gender/add:0', 'softmax_smile/add:0', 'softmax_age/add:0'), feed_dict={'Placeholder:0':feature})
	        for i in xrange(4):
	            if i < 3:
	    	        result.append(np.argmax(score[i], axis=1))
	    	else:
	    	    result.append(np.squeeze( (score[i]+1.0)*50.0))
	
	return result

    def destroy(self):
        self.sess.close()
