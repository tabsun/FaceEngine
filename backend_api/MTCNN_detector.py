import tensorflow as tf
import tools
import cv2, os
import numpy as np
from mtcnn import PNet, RNet, ONet
import datetime

class MTCNN_detector:
    def __init__(self, model_path, gpu_id=0, minsize=40, factor=0.5, threshold=[0.8,0.8,0.9]):
        rnet_model_path = os.path.join(model_path, "rnet/rnet-3000000")
        pnet_model_path = os.path.join(model_path, "pnet/pnet-3000000")
        onet_model_path = os.path.join(model_path, "onet/onet-500000")
        if not os.path.exists(model_path) or \
            not os.path.exists(os.path.join(model_path,"rnet")) or \
            not os.path.exists(os.path.join(model_path,"pnet")) or \
            not os.path.exists(os.path.join(model_path,"onet")):
            raise Exception("Error when loading {}".format(model_path))
        # default detection parameters
        self.minsize = minsize
        self.factor = factor
        self.threshold = threshold
        # load models
        with tf.device('/gpu:{}'.format(gpu_id)):
            with tf.Graph().as_default() as p:
                config = tf.ConfigProto(allow_soft_placement=True)
                self.sess = tf.Session(config=config)
                self.pnet_input = tf.placeholder(tf.float32, [None,None,None,3])
                self.pnet = PNet({'data':self.pnet_input}, mode='test')
                self.pnet_output = self.pnet.get_all_output()

                self.rnet_input = tf.placeholder(tf.float32, [None,24,24,3])
                self.rnet = RNet({'data':self.rnet_input}, mode='test')
                self.rnet_output = self.rnet.get_all_output()

                self.onet_input = tf.placeholder(tf.float32, [None,48,48,3])
                self.onet = ONet({'data':self.onet_input}, mode='test')
                self.onet_output = self.onet.get_all_output()

                saver_pnet = tf.train.Saver([v for v in tf.global_variables() if v.name[0:5] == "pnet/"])
                saver_rnet = tf.train.Saver([v for v in tf.global_variables() if v.name[0:5] == "rnet/"])
                saver_onet = tf.train.Saver([v for v in tf.global_variables() if v.name[0:5] == "onet/"])

                saver_pnet.restore(self.sess, pnet_model_path)
                self.pnet_func = lambda img: self.sess.run(self.pnet_output, feed_dict={self.pnet_input:img})
                saver_rnet.restore(self.sess, rnet_model_path)
                self.rnet_func = lambda img: self.sess.run(self.rnet_output, feed_dict={self.rnet_input:img})
                saver_onet.restore(self.sess, onet_model_path)
                self.onet_func = lambda img: self.sess.run(self.onet_output, feed_dict={self.onet_input:img})

    def destroy(self):
        self.sess.close()
    
    # Returns:
    #     rects: a numpy array of shape [num_face, 5].
    #            Denote of each row:
    #            [left_top_x, left_top_y, right_bottom_x, right_bottom_y, confidence] 
    def calc_det_result(self, image):
        rects, shapes = self.calc_landmark_result(image)
        return rects

    # Returns:
    #     rectangles: a numpy array of shape [num_face, 5].
    #                 Denote of each row:
    #                 [left_top_x, left_top_y, right_bottom_x, right_bottom_y, confidence] 
    #     points:     a numpy array of shape [num_face, 10], 
    #                 Denote of each row:
    #                 [left_eye_x, left_eye_y, right_eye_x, right_eye_y, 
    #                 nose_x, nose_y,
    #                 left_mouthcorner_x,left_mouthcorner_y,right_mouthcorner_x,right_mouthcorner_y,]
    def calc_landmark_result(self, image):
        # TODO time test
        start = cv2.getTickCount()
        rectangles, shapes = tools.detect_face(image, self.minsize, self.pnet_func, self.rnet_func, self.onet_func, self.threshold, self.factor)
        shapes = np.transpose(shapes)
        # TODO time test
        usetime = (cv2.getTickCount() - start)/cv2.getTickFrequency()
        print "Use time {}s.".format(usetime)
        return rectangles, shapes
    
    # SAME as calc_landmark_result but return shape [num_face, 4] 
    # Denote of each row:
    #       [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
    def extract_eye_result(self, shapes):
        assert(shapes is not None)
        assert(shapes.shape[0] > 0 and shapes.shape[1] == 10)
        return shapes[:,0:4]
    
    # show the detection results
    def show_result(self, image_path, rectangles, shapes):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if rectangles.shape[0] != shapes.shape[0]:
            print "Error in show results {} != {}.".format(rectangles.shape[0], shapes.shape[0])
        for rect in rectangles:
            cv2.rectangle(image, (int(round(rect[0])),int(round(rect[1]))), \
            (int(round(rect[2])),int(round(rect[3]))), (255,255,0), 2)
        for shape in shapes:
            shape_num = len(shape) / 2
            for i in xrange(shape_num):
                pt = (int(round(shape[2*i])),int(round(shape[2*i+1])))
                cv2.circle(image, pt, 2, (0,0,255), 2)
        cv2.imwrite("show.jpg",image)
