#facenet.py
import tensorflow as tf 
import pickle
import numpy as np
import os
from FACENET_alignmentor import FACENET_alignmentor 

class FACENET_recognizor:
    def __init__(self, model_path, gpu_id=0, dims=[128, 768], batchsize=1):
	self.alignmentor = FACENET_alignmentor()
	with tf.device('/gpu:{}'.format(gpu_id)):
	    file_path = os.path.join(model_path, "facenet_model.pkl")
	    if(not os.path.exists(file_path)):
		raise Exception("{} do not exists.".format(file_path))
	    self.face = facenet(file_path)
	    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
	    init = tf.global_variables_initializer()
	    self.sess = tf.Session(config=config)
	    self.sess.run(init)
	    self.dims = dims
	    self.batchsize = batchsize
	    self.input_nparray = np.zeros((batchsize, 128, 128, 3), dtype=np.float)
    # ug_images : the unaligned images to extract features
    # shapes    : the corresponing eye points for each detected face in images
    # layer_ids : 0 - output 128-dim feature
    #             1 - baoming's 640-dim feature
    # if you want both you should set layer_ids = [0,1]
    # if you want [baoming's feature, 128-dim output], then you should set layer_ids = [1, 0]
    def get_feature(self, ug_images, shapes, layer_ids):
	assert(ug_images.shape[0] == shapes.shape[0])
	images = []
        for i in xrange(len(ug_images)):
	    images.append(self.alignmentor.get_result(ug_images[i], shapes[i]))
	actualsize = len(images)
	images = np.array(images)
        features = np.array([])
	if True:
	    assert( set(images.shape[1:4]) == set([128,128,3]) )
	    accum_num = 0
	    while(accum_num < actualsize):
	        cur_num = self.batchsize if actualsize-accum_num >= self.batchsize else actualsize-accum_num
	        self.input_nparray[0:cur_num,:,:,:] = images[accum_num:accum_num+cur_num,:,:,:]
	        feat_list = self.sess.run( self.face.output, {self.face.input:self.input_nparray})
	        assert(len(layer_ids) <= len(feat_list))
	        cur_feature = np.array([])
	        for id in layer_ids:
	    	    feature = feat_list[id][0:cur_num,:].reshape((cur_num, -1))
	    	    cur_feature = np.hstack([cur_feature, feature]) if cur_feature.size else feature
	        
	        features = np.vstack([features, cur_feature]) if features.size else cur_feature
	        accum_num += cur_num
	    assert(features.shape[0] == actualsize)
	assert( features.shape[1] in self.dims )	
	return features
    
    def get_score(self, a, b):
        assert(np.prod(a.shape) == np.prod(b.shape))
	assert(np.prod(a.shape) in self.dims)
        s = np.dot(a.reshape((1,-1)),b.reshape((-1,1)))/(np.linalg.norm(a) * np.linalg.norm(b))
        if( s<0.5 ):
            s = (s + 1.0)/3.0
        return s

    def destroy(self):
	self.sess.close()

class facenet:
    def __init__(self, model_path):
	assert(os.path.exists(model_path))
        with open(model_path, 'rb') as f:
            self.params = pickle.load(f)
            keys = self.params.keys()
            keys.sort()
            #for key in keys:
            #    print key ," with shape ", self.params[key].shape
        self.input = tf.placeholder(tf.float32, shape=(None,128,128,3))
	self.output = self.net(self.input)

    
    def _get_variable(self, iname, ishape, inverse=False):
        assert(self.params.has_key(iname))
        weight_array = self.params[iname]
        
        if len(ishape) == 1:
            reshape_weight_array = weight_array.reshape(tuple(ishape))
        elif len(ishape) == 2:
            reshape_weight_array = np.zeros(ishape,np.float)
            in_num,out_num = ishape[:]
            in_num_,w_,h_,out_num_ = weight_array.shape[:]
            assert(w_==1 and h_==1 and in_num==in_num_ and out_num==out_num_)
            for n in xrange(in_num):
                for m in xrange(out_num):
                    reshape_weight_array[n,m] = weight_array[n,0,0,m]
        else:
            # the tensor in params has dim: in_channels x width x height x out_channels
            # but here tensorflow has dim : height x width x in_channles x out_channels
            reshape_weight_array = np.zeros(ishape, np.float)
            h,w,in_num,out_num = ishape[:]
            in_num_,w_,h_,out_num_ = weight_array.shape[:]
            assert(h==h_ and w==w_ and in_num==in_num_ and out_num==out_num_)
            for i in xrange(h):
                for j in xrange(w):
                    for n in xrange(in_num):
                        for m in xrange(out_num):
                            reshape_weight_array[i,j,n,m] = weight_array[n,j,i,m]

        #reshape_weight_array = weight_array.reshape(tuple(ishape))
        if inverse:
            reshape_weight_array = 1.0 / reshape_weight_array
        assert(reshape_weight_array.shape == tuple(ishape))
        return tf.get_variable(name=iname, dtype=tf.float32, initializer=tf.constant(reshape_weight_array,dtype=tf.float32))

    def _get_variable2(self, iname, ishape, inverse=False):
        assert(self.params.has_key(iname))
        weight_array = self.params[iname]
        return tf.get_variable(name=iname, dtype=tf.float32, initializer=tf.constant(weight_array,dtype=tf.float32))

    def _get_special_variable(self, iname, ishape):
        assert(self.params.has_key(iname))
        weight_array = self.params[iname]
        
        # the tensor in params has dim: in_channels x width x height x out_channels
        # but here tensorflow has dim : height x width x in_channles x out_channels
        reshape_weight_array = np.zeros(ishape, np.float)
        in_num,out_num = ishape[:]
        in_num_,w_,h_,out_num_ = weight_array.shape[:]
        assert(in_num_*w_*h_*out_num_ == out_num*in_num)
        for i in xrange(h_):
            for j in xrange(w_):
                for n in xrange(in_num_):
                    for m in xrange(out_num_):
                        idx = n + j*in_num_ + i*w_*in_num_
                        reshape_weight_array[idx,m] = weight_array[n,j,i,m]

        #reshape_weight_array = np.transpose(reshape_weight_array)
        assert(reshape_weight_array.shape == tuple(ishape))
        return tf.get_variable(name=iname, dtype=tf.float32, initializer=tf.constant(reshape_weight_array,dtype=tf.float32))

    def _batch_norm(self, x, mean, multiplier, offset):          
        return tf.nn.batch_normalization(x, mean, 1.0, offset, multiplier, 1e-5) 

    def _res_module(self, x, idx):            
        with tf.variable_scope("res_m_%d" % idx):
            inc_0_5x5_proj_w=  self._get_variable("inc_%d_5x5_proj_w"%idx, [1,1,96,80])
            inc_0_5x5_proj_b= self._get_variable("inc_%d_5x5_proj_b"%idx, [80])
            inc_0_5x5_proj = tf.nn.conv2d(x, inc_0_5x5_proj_w, strides=[1,1,1,1], padding = 'SAME')
            inc_0_5x5_proj = tf.nn.bias_add(inc_0_5x5_proj, inc_0_5x5_proj_b)
            inc_0_5x5_proj_act = tf.nn.relu6(inc_0_5x5_proj) 
            inc_0_5x5_depthwise_w = self._get_variable("inc_%d_5x5_depthwise_w"%idx, [5,5,80,1])
            inc_0_5x5_depthwise = tf.nn.depthwise_conv2d(inc_0_5x5_proj_act, inc_0_5x5_depthwise_w, strides=[1,1,1,1], padding='SAME')
            mean = self._get_variable("inc_%d_5x5_bn_mean"%idx, [80])
            multiplier = self._get_variable("inc_%d_5x5_bn_multiplier"%idx, [80], inverse=False)
            offset = self._get_variable("inc_%d_5x5_bn_offset"%idx, [80])
            inc_0_5x5_bn = self._batch_norm(inc_0_5x5_depthwise, mean, multiplier, offset)
            inc_0_5x5_act = tf.nn.relu6(inc_0_5x5_bn) 

            inc_0_concat = tf.concat(inc_0_5x5_act,axis=3) 

            inc_0_up_w = self._get_variable("inc_%d_up_w"%idx, [1,1,80,96])
            inc_0_up_b = self._get_variable("inc_%d_up_b"%idx, [96])
            inc_0_up = tf.nn.conv2d(inc_0_concat, inc_0_up_w,strides=[1,1,1,1],padding='SAME') 
            inc_0_up = tf.nn.bias_add(inc_0_up, inc_0_up_b)

            nc_0_add = tf.add(inc_0_up,x) 
            inc_0_activation = tf.nn.relu6(nc_0_add) 

            return inc_0_activation

    def _res_module2(self, x, idx):            
        with tf.variable_scope("res_m2_%d" % idx):
            last_0_5x5_proj_w =  self._get_variable("last_%d_5x5_proj_w"%idx, [1,1,160,80])
            last_0_5x5_proj_b = self._get_variable("last_%d_5x5_proj_b"%idx, [80])
            last_0_5x5_proj = tf.nn.conv2d(x, last_0_5x5_proj_w, strides=[1,1,1,1], padding = 'SAME')
            last_0_5x5_proj = tf.nn.bias_add(last_0_5x5_proj, last_0_5x5_proj_b)
            last_0_5x5_proj_act = tf.nn.relu6(last_0_5x5_proj)
            last_0_5x5_depthwise_w = self._get_variable("last_%d_5x5_depthwise_w"%idx, [5,5,80,1])
            last_0_5x5_depthwise = tf.nn.depthwise_conv2d(last_0_5x5_proj_act, last_0_5x5_depthwise_w, strides=[1,1,1,1], padding='SAME')

            mean = self._get_variable("last_%d_5x5_bn_mean"%idx, [80])
            multiplier = self._get_variable("last_%d_5x5_bn_multiplier"%idx, [80], inverse=False)
            offset = self._get_variable("last_%d_5x5_bn_offset"%idx, [80])
            last_0_5x5_bn = self._batch_norm(last_0_5x5_depthwise, mean, multiplier, offset)
            last_0_5x5_act = tf.nn.relu6(last_0_5x5_bn)

            last_0_concat = tf.concat(last_0_5x5_act,axis=3)

            last_0_up_w = self._get_variable("last_%d_up_w"%idx, [1,1,80,160])
            last_0_up_b = self._get_variable("last_%d_up_b"%idx, [160])
            last_0_up = tf.nn.conv2d(last_0_concat, last_0_up_w, strides=[1,1,1,1],padding='SAME')
            last_0_up = tf.nn.bias_add(last_0_up, last_0_up_b)

            last_0_add = tf.add(last_0_up,x)
            last_0_activation = tf.nn.relu6(last_0_add)

            return last_0_activation

    def _local_module(self, x, idx):
        with tf.variable_scope("local_m_%d" % idx):
            fc_1_fc_fc_w = self._get_variable("fc_%d_fc_fc_w"%idx, [512,512])
            fc_1_fc_fc = tf.matmul(x, fc_1_fc_fc_w)
            mean = self._get_variable("fc_%d_fc_mean"%idx, [512])
            multiplier = self._get_variable("fc_%d_fc_multiplier"%idx, [512], inverse=False)
            offset = self._get_variable("fc_%d_fc_offset"%idx, [512])
            fc_1_fc = self._batch_norm(fc_1_fc_fc, mean, multiplier, offset)
            fc_1_add = tf.add(x, fc_1_fc)
            fc_1 = tf.nn.relu6(fc_1_add)
            return fc_1

    def net(self, input):
        self.conv_0_w = self._get_variable("conv_0_w", [5,5,3,32])
        self.conv_0_b = self._get_variable("conv_0_b", [32])
        conv_0 = tf.nn.conv2d(input, self.conv_0_w, strides=[1, 2, 2, 1], padding='SAME')
        conv_0 = tf.nn.bias_add(conv_0,self.conv_0_b)
        conv_0_act = tf.nn.relu6(conv_0)
        self.downsample_0_3x3_proj_w = self._get_variable("downsample_0_3x3_proj_w", [1,1,32,32])
        self.downsample_0_3x3_proj_b = self._get_variable("downsample_0_3x3_proj_b", [32])
        downsample_0_3x3_proj = tf.nn.conv2d(conv_0_act, self.downsample_0_3x3_proj_w, strides=[1, 1, 1, 1], padding='SAME')
        downsample_0_3x3_proj = tf.nn.bias_add(downsample_0_3x3_proj,self.downsample_0_3x3_proj_b)
        downsample_0_3x3_proj_act = tf.nn.relu6(downsample_0_3x3_proj)
        self.downsample_0_3x3_depthwise_w = self._get_variable("downsample_0_3x3_depthwise_w", [3,3,32,1])
        downsample_0_3x3_depthwise = tf.nn.depthwise_conv2d(downsample_0_3x3_proj_act, self.downsample_0_3x3_depthwise_w, strides=[1,2,2,1], padding='SAME')

        self.downsample_0_3x3_bn_mean = self._get_variable("downsample_0_3x3_bn_mean", [32])
        self.downsample_0_3x3_bn_multiplier = self._get_variable("downsample_0_3x3_bn_multiplier", [32], inverse=False)
        self.downsample_0_3x3_bn_offset = self._get_variable("downsample_0_3x3_bn_offset", [32])

        downsample_0_3x3_bn = self._batch_norm(downsample_0_3x3_depthwise, self.downsample_0_3x3_bn_mean, \
                self.downsample_0_3x3_bn_multiplier, self.downsample_0_3x3_bn_offset)
        
        downsample_0_3x3_act = tf.nn.relu6(downsample_0_3x3_bn)
        downsample_0_concat = tf.concat([downsample_0_3x3_act], axis=3)

        self.downsample_0_up_w = self._get_variable("downsample_0_up_w", [1,1,32,48])
        self.downsample_0_up_b = self._get_variable("downsample_0_up_b", [48])

        downsample_0_up = tf.nn.conv2d(downsample_0_concat, self.downsample_0_up_w,strides=[1,1,1,1],padding='VALID')
        downsample_0_up = tf.nn.bias_add(downsample_0_up, self.downsample_0_up_b)
        downsample_0_activation = tf.nn.relu6(downsample_0_up)

        # downsample_0_activation-downsample_1_activation
        # downsample_1_3x3
        self.downsample_1_3x3_proj_w = self._get_variable("downsample_1_3x3_proj_w", [1,1,48,64])
        self.downsample_1_3x3_proj_b = self._get_variable("downsample_1_3x3_proj_b", [64])
        downsample_1_3x3_proj = tf.nn.conv2d(downsample_0_activation, self.downsample_1_3x3_proj_w, strides=[1,1,1,1],padding='SAME')
        downsample_1_3x3_proj = tf.nn.bias_add(downsample_1_3x3_proj, self.downsample_1_3x3_proj_b)
        downsample_1_3x3_proj_act = tf.nn.relu6(downsample_1_3x3_proj)

        self.downsample_1_3x3_depthwise_w = self._get_variable("downsample_1_3x3_depthwise_w", [3,3,64,1])
        downsample_1_3x3_depthwise = tf.nn.depthwise_conv2d(downsample_1_3x3_proj_act, self.downsample_1_3x3_depthwise_w, strides=[1,2,2,1], padding='SAME')
                 
        self.downsample_1_3x3_bn_mean = self._get_variable("downsample_1_3x3_bn_mean", [64])
        self.downsample_1_3x3_bn_multiplier = self._get_variable("downsample_1_3x3_bn_multiplier", [64], inverse=False)
        self.downsample_1_3x3_bn_offset = self._get_variable("downsample_1_3x3_bn_offset", [64])

        downsample_1_3x3_bn = self._batch_norm(downsample_1_3x3_depthwise,self.downsample_1_3x3_bn_mean, \
                self.downsample_1_3x3_bn_multiplier,self.downsample_1_3x3_bn_offset)
        downsample_1_3x3_act = tf.nn.relu6(downsample_1_3x3_bn) 
        # downsample_1_5x5
        self.downsample_1_5x5_proj_w = self._get_variable("downsample_1_5x5_proj_w", [1,1,48,32])
        self.downsample_1_5x5_proj_b = self._get_variable("downsample_1_5x5_proj_b", [32])
        downsample_1_5x5_proj = tf.nn.conv2d(downsample_0_activation, self.downsample_1_5x5_proj_w, strides=[1,1,1,1], padding='SAME') 
        downsample_1_5x5_proj = tf.nn.bias_add(downsample_1_5x5_proj, self.downsample_1_5x5_proj_b)
        downsample_1_5x5_proj_act = tf.nn.relu6(downsample_1_5x5_proj) 

        self.downsample_1_5x5_depthwise_w = self._get_variable("downsample_1_5x5_depthwise_w", [5,5,32,1]) 
        downsample_1_5x5_depthwise = tf.nn.depthwise_conv2d(downsample_1_5x5_proj_act, self.downsample_1_5x5_depthwise_w, strides=[1,2,2,1], padding='SAME') 
                  
        self.downsample_1_5x5_bn_mean = self._get_variable("downsample_1_5x5_bn_mean", [32]) 
        self.downsample_1_5x5_bn_multiplier = self._get_variable("downsample_1_5x5_bn_multiplier", [32], inverse=False) 
        self.downsample_1_5x5_bn_offset = self._get_variable("downsample_1_5x5_bn_offset", [32]) 

        downsample_1_5x5_bn = self._batch_norm(downsample_1_5x5_depthwise, self.downsample_1_5x5_bn_mean, \
                self.downsample_1_5x5_bn_multiplier, self.downsample_1_5x5_bn_offset)
        
        downsample_1_5x5_act = tf.nn.relu6(downsample_1_5x5_bn) 

        # Merge downsample_1_xxx_act
        downsample_1_concat = tf.concat([downsample_1_3x3_act, downsample_1_5x5_act], axis = 3) 

        self.downsample_1_up_w = self._get_variable("downsample_1_up_w", [1,1,96,96]) 
        self.downsample_1_up_b = self._get_variable("downsample_1_up_b", [96]) 
        downsample_1_up = tf.nn.conv2d(downsample_1_concat, self.downsample_1_up_w, strides = [1,1,1,1], padding='SAME') 
        downsample_1_up = tf.nn.bias_add(downsample_1_up, self.downsample_1_up_b) 
        downsample_1_activation = tf.nn.relu6(downsample_1_up) 

        # loop inc_x_activation
        inc_0_activation = self._res_module(downsample_1_activation,0) 
        inc_1_activation = self._res_module(inc_0_activation,1) 
        inc_2_activation = self._res_module(inc_1_activation,2) 
        inc_3_activation = self._res_module(inc_2_activation,3) 
        inc_4_activation = self._res_module(inc_3_activation,4) 
        inc_5_activation = self._res_module(inc_4_activation,5) 
        inc_6_activation = self._res_module(inc_5_activation,6) 
        inc_7_activation = self._res_module(inc_6_activation,7) 
        inc_8_activation = self._res_module(inc_7_activation,8) 
        inc_9_activation = self._res_module(inc_8_activation,9) 
        inc_10_activation = self._res_module(inc_9_activation,10) 
        inc_11_activation = self._res_module(inc_10_activation,11) 
        inc_12_activation = self._res_module(inc_11_activation,12) 
        inc_13_activation = self._res_module(inc_12_activation,13) 

        # inc_13_activation~downsample_2_activation
        downsample_2_5x5_proj_w = self._get_variable("downsample_2_5x5_proj_w", [1,1,96,160]) 
        downsample_2_5x5_proj_b = self._get_variable("downsample_2_5x5_proj_b", [160]) 

        downsample_2_5x5_proj = tf.nn.conv2d(inc_13_activation, downsample_2_5x5_proj_w, strides=[1,1,1,1], padding='VALID') 
        downsample_2_5x5_proj = tf.nn.bias_add(downsample_2_5x5_proj, downsample_2_5x5_proj_b) 
	# Here baoming add one another feature
	baoming_feat = tf.nn.avg_pool(downsample_2_5x5_proj, ksize=[1,8,8,1], strides=[1,8,8,1], padding='VALID')
	baoming_feat_norm = tf.nn.l2_normalize(baoming_feat, dim=3)
	baoming_feat_norm = tf.reshape(baoming_feat_norm, [-1, 640])

        downsample_2_5x5_proj_act = tf.nn.relu6(downsample_2_5x5_proj) 

        downsample_2_5x5_depthwise_w = self._get_variable("downsample_2_5x5_depthwise_w", [5,5,160,1]) 
        downsample_2_5x5_depthwise = tf.nn.depthwise_conv2d(downsample_2_5x5_proj_act, downsample_2_5x5_depthwise_w, strides=[1,2,2,1], padding='SAME') 

        downsample_2_5x5_bn_mean = self._get_variable("downsample_2_5x5_bn_mean", [160]) 
        downsample_2_5x5_bn_multiplier = self._get_variable("downsample_2_5x5_bn_multiplier", [160], inverse=False) 
        downsample_2_5x5_bn_offset = self._get_variable("downsample_2_5x5_bn_offset", [160]) 

        downsample_2_5x5_bn = self._batch_norm(downsample_2_5x5_depthwise, downsample_2_5x5_bn_mean, \
                downsample_2_5x5_bn_multiplier, downsample_2_5x5_bn_offset)
                 
        downsample_2_5x5_act = tf.nn.relu6(downsample_2_5x5_bn) 

        downsample_2_concat = tf.concat([downsample_2_5x5_act],axis = 3) 
        downsample_2_up_w = self._get_variable("downsample_2_up_w", [1,1,160,160]) 
        downsample_2_up_b = self._get_variable("downsample_2_up_b", [160]) 

        downsample_2_up = tf.nn.conv2d(downsample_2_concat, downsample_2_up_w, strides=[1,1,1,1], padding='SAME') 
        downsample_2_up = tf.nn.bias_add(downsample_2_up, downsample_2_up_b) 

        downsample_2_activiation = tf.nn.relu6(downsample_2_up) 
        
        # Last loop
        last_0_activation = self._res_module2(downsample_2_activiation,0) 
        last_1_activation = self._res_module2(last_0_activation,1) 
        last_2_activation = self._res_module2(last_1_activation,2) 
        last_3_activation = self._res_module2(last_2_activation,3) 
        
        # last_3_activation~downsample_3_activation
        downsample_3_5x5_proj_w = self._get_variable("downsample_3_5x5_proj_w", [1,1,160,128]) 
        downsample_3_5x5_proj_b = self._get_variable("downsample_3_5x5_proj_b", [128]) 
        downsample_3_5x5_proj = tf.nn.conv2d(last_3_activation, downsample_3_5x5_proj_w, strides = [1,1,1,1], padding = 'SAME') 
        downsample_3_5x5_proj = tf.nn.bias_add(downsample_3_5x5_proj,downsample_3_5x5_proj_b) 
        downsample_3_5x5_proj_act = tf.nn.relu6(downsample_3_5x5_proj) 

        downsample_3_5x5_depthwise_w = self._get_variable("downsample_3_5x5_depthwise_w", [5,5,128,1]) 
        downsample_3_5x5_depthwise = tf.nn.depthwise_conv2d(downsample_3_5x5_proj_act, downsample_3_5x5_depthwise_w, strides = [1,2,2,1], padding='SAME') 

        downsample_3_5x5_bn_mean = self._get_variable("downsample_3_5x5_bn_mean", [128]) 
        downsample_3_5x5_bn_multiplier = self._get_variable("downsample_3_5x5_bn_multiplier", [128], inverse=False) 
        downsample_3_5x5_bn_offset = self._get_variable("downsample_3_5x5_bn_offset", [128]) 

        downsample_3_5x5_bn = self._batch_norm(downsample_3_5x5_depthwise, downsample_3_5x5_bn_mean, \
                downsample_3_5x5_bn_multiplier, downsample_3_5x5_bn_offset)
        downsample_3_5x5_act = tf.nn.relu6(downsample_3_5x5_bn) 

        downsample_3_concat = tf.concat([downsample_3_5x5_act], axis=3) 

        downsample_3_up_w = self._get_variable("downsample_3_up_w", [1,1,128,128]) 
        downsample_3_up_b = self._get_variable("downsample_3_up_b", [128]) 

        downsample_3_up = tf.nn.conv2d(downsample_3_concat, downsample_3_up_w,strides=[1,1,1,1], padding = "SAME") 
        downsample_3_up = tf.nn.bias_add(downsample_3_up, downsample_3_up_b) 
        downsample_3_activation = tf.nn.relu6(downsample_3_up) 

        # downsample_3_activation~fc_0
        conv_fc_proj_w = self._get_variable("conv_fc_proj_w", [3,3,128,128]) 
        conv_fc_proj_b = self._get_variable("conv_fc_proj_b", [128]) 
        conv_fc_proj = tf.nn.conv2d(downsample_3_activation, conv_fc_proj_w, strides=[1,1,1,1], padding = "VALID") 
        conv_fc_proj = tf.nn.bias_add(conv_fc_proj, conv_fc_proj_b) 
        conv_fc_proj_act = tf.nn.relu6(conv_fc_proj) 

        # here you may use transpose to replace process in _get_variable
        fc_0_fc_w = self._get_variable("fc_0_fc_w", [2,2,128,512]) 
        #fc_0_fc_w_t = tf.reshape(tf.transpose(fc_0_fc_w, perm=[2,1,0,3]),[512,512])
        fc_0_fc_w_t = tf.reshape(fc_0_fc_w, [512,512])
        conv_fc_t = tf.reshape(conv_fc_proj_act,[-1,512])
        fc_0_fc = tf.matmul(conv_fc_t, fc_0_fc_w_t) 

        fc_0_bn_mean = self._get_variable("fc_0_bn_mean", [512]) 
        fc_0_bn_multiplier = self._get_variable("fc_0_bn_multiplier", [512], inverse=False) 
        fc_0_bn_offset = self._get_variable("fc_0_bn_offset", [512]) 
        fc_0_bn = self._batch_norm(fc_0_fc, fc_0_bn_mean, fc_0_bn_multiplier, fc_0_bn_offset)
        fc_0 = tf.nn.relu6(fc_0_bn) 

        # FC loop
        fc_1 = self._local_module(fc_0,1) 
        fc_2 = self._local_module(fc_1,2) 
        fc_3 = self._local_module(fc_2,3) 
        # LOG REGION
        #print "fc_0_fc_w"
        #print np.mean(np.absolute(self.params["fc_0_fc_w"]))
        #print "fc_1_fc_fc_w"
        #print np.mean(self.params["fc_1_fc_fc_w"])
        #print "fc_2_fc_fc_w"
        #print np.mean(self.params["fc_2_fc_fc_w"])
        #print "fc_3_fc_fc_w"
        #print np.mean(self.params["fc_3_fc_fc_w"])

        #print "fc_0_bn_multiplier"
        #print np.mean(self.params["fc_0_bn_multiplier"])
        #print "fc_1_bn_multiplier"
        #print np.mean(self.params["fc_1_fc_multiplier"])
        #print "fc_2_bn_multiplier"
        #print np.mean(self.params["fc_2_fc_multiplier"])
        #print "fc_3_bn_multiplier"
        #print np.mean(self.params["fc_3_fc_multiplier"])
        
        #print self.params["fc_0_fc_w"]
        # fc_3~L2
        projection_w = self._get_variable("projection_w", [512,128]) 
        projection_b = self._get_variable("projection_b", [128]) 
        proj = tf.matmul(fc_3,projection_w) 
        proj = tf.nn.bias_add(proj, projection_b) 
        projection = tf.nn.l2_normalize(proj, dim=1) 
        # TODO: maybe after L2_norm the net will further process the feature or the feature is not extracted from this layer
        # feature = projection * 128
        return [projection, baoming_feat_norm]

