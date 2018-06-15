import numpy as np
import cv2, nudged, os

class LANDMARK_ajuster:
    def __init__(self, model_path, type_str="tensorflow_106"):
        self.RefPts = np.array('35, 32, 77, 32, 56, 59, 39, 79, 73, 79'.strip().split(',')).reshape(-1,2).astype(float)
        if(type_str == "tensorflow_106"):
            global tf
            import tensorflow as tf
            from FACENET_alignmentor import FACENET_alignmentor
	    model_name = os.path.join( model_path, "ajuster_model/alignment" )
            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.allow_growth = True
            #config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1))
            self.sess = tf.Session(config=config)

            self.alignmentor = FACENET_alignmentor()
            saver = tf.train.import_meta_graph(model_name+'.meta', import_scope='align')
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, model_name)
            self.crop_size = 128
            self.input_size = 128
	    self.type_str = type_str
 
        if(type_str == "caffe_106"):
            global caffe
            import caffe
            assert(os.path.exists(os.path.join(model_path, "ajuster_model/106pts_net1.deploy")))
            assert(os.path.exists(os.path.join(model_path, "ajuster_model/106pts_net1.caffemodel")))
            self.net = caffe.Net(os.path.join(model_path, 'ajuster_model/106pts_net1.deploy'), \
                                 os.path.join(model_path, 'ajuster_model/106pts_net1.caffemodel'), \
                                 caffe.TEST) 
            self.crop_size = 112 
            self.input_size = 56
            self.net.blobs['data'].reshape(1, 3, self.input_size, self.input_size)
            tmp_img = cv2.imread(os.path.join(model_path, "test.jpg"))
            tmp_img = cv2.resize(tmp_img, (self.input_size, self.input_size))
            input = np.asarray([tmp_img.transpose(2,0,1) - 128.0])
            self.net.blobs['data'].data[...] = input 
            self.net.forward()
            self.type_str = type_str

    def extract_five_result(self, shapes):
        assert(shapes is not None)
        assert(shapes.shape[0] > 0)
        five_shapes = []
        for shape in shapes:
            five_shape = []
            if(shapes.shape[1] == 136):
                five_shape.append( round((shape[37*2] + shape[38*2] + shape[40*2] + shape[41*2])/4.0) )
                five_shape.append( round((shape[37*2+1] + shape[38*2+1] + shape[40*2+1] + shape[41*2+1])/4.0) )
                five_shape.append( round((shape[43*2] + shape[44*2] + shape[46*2] + shape[47*2])/4.0) )
                five_shape.append( round((shape[43*2+1] + shape[44*2+1] + shape[46*2+1] + shape[47*2+1])/4.0) )
    	        five_shape.append( round(shape[33*2]) )
    	        five_shape.append( round(shape[33*2+1]) )
    	        five_shape.append( round(shape[48*2]) )
    	        five_shape.append( round(shape[48*2+1]) )
                five_shape.append( round(shape[54*2]) )
    	        five_shape.append( round(shape[54*2+1]) )               
            if(shapes.shape[1] == 10):
                five_shape = shape
            five_shapes.append(five_shape)
        return np.array(five_shapes)

    # image - the whole image
    # shapes - n x 10 in which each row denotes left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, ..._y, left_mouthcorner_x,...y, right_..._x, ..._y
    # return - n x 212 in which each row denotes x0,y0,x1,y1...
    def calc_106_landmarks(self, image, shapes):
        images = []
        tform_reverses = []
        results = []
	# preprocess and alignment
        for shape in shapes:
            shape = shape.reshape(-1, 2)
            tform = nudged.estimate(shape, self.RefPts)
            tform = np.asarray(tform.get_matrix())

            img_warp = cv2.warpPerspective(image.copy(), tform, (self.crop_size,self.crop_size))
            img_tmp = cv2.resize(img_warp,(self.input_size,self.input_size))
            if(self.type_str == "caffe_106"):
                img_tmp = img_tmp.transpose(2,0,1) - 128.0
            else:
                img_tmp = (img_tmp - 127.5) * (1. / 128.0)
            tform_reverses.append(np.asarray(np.mat(tform).I))
            images.append(img_tmp)

        if(len(images) == 0):
	    return results
        images = np.asarray(images)
														        
        if(self.type_str == "tensorflow_106"):
            outputs = self.sess.run('align/fc2/fc2:0', feed_dict={'align/alignment_input:0':images})
            for index,(output, tform_reverse) in enumerate(zip(outputs, tform_reverses)):
                point_aligned = np.concatenate((output.reshape(-1, 2) * (self.crop_size*1.0 / self.input_size), \
                                  np.ones((106,1),dtype=np.float32)),axis=1)
                point_original = np.dot(tform_reverse, point_aligned.T)[:2,:].T
	        point_original = point_original.reshape((-1,1))
	        results.append(point_original)
        else:
            for index, (image, tform_reverse) in enumerate(zip(images, tform_reverses)):
                self.net.blobs['data'].data[...] = np.asarray([image])
                self.net.forward()
                output = (self.net.blobs['fc106pts'].data[...].reshape(-1,2)+self.input_size/2) * (self.crop_size*1.0/self.input_size)
                point_aligned = np.concatenate((output.reshape(-1, 2), np.ones((106,1),dtype=np.float32)),axis=1)
                point_original = np.dot(tform_reverse, point_aligned.T)[:2,:].T
                point_original = point_original.reshape((-1,1))
                results.append(point_original)

        return np.around(np.array( results ))
