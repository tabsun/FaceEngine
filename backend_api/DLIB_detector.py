import cv2, dlib, os, math
import numpy as np

class DLIB_detector:
    def __init__(self, model_path, minsize=40):
        model_path = os.path.join(model_path, "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(model_path):
            raise Exception( "Error when loading {}".format(model_path) )
        # default detection parameters
        self.minsize = minsize
        # load models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    # Returns:
    #     rectangles: a numpy array of shape [num_face, 5].
    #                 Denote of each row:
    #                 [left_top_x, left_top_y, right_bottom_x, right_bottom_y, confidence] 
    def calc_det_result(self, image):
        level = max(0, int(math.ceil(math.log(80.0/self.minsize, 2))))
        dets, scores, idx = self.detector.run(image, level)
        rectangles = [[dets[i].left(), dets[i].top(), dets[i].right(), dets[i].bottom(), scores[i]] \
         for i in xrange(len(dets))]
        return rectangles

    # Returns:
    #     rectangles: a numpy array of shape [num_face, 5].
    #                 Denote of each row:
    #                 [left_top_x, left_top_y, right_bottom_x, right_bottom_y, confidence] 
    #     points:     a numpy array of shape [num_face, 136], 
    #                 Denote of each row:
    #                 [x0, y0, x1, y1, x2, y2...,x135,y135]
    def calc_landmark_result(self, image):
        # TODO time test
        start = cv2.getTickCount()
        level = max(0, int(math.ceil(math.log(80.0/self.minsize, 2)))) 
        dets, scores, idx = self.detector.run(image, level)
        rectangles = [[dets[i].left(), dets[i].top(), dets[i].right(), dets[i].bottom(), scores[i]] \
         for i in xrange(len(dets))]
        # Get the shapes from face detection results
        shapes = []
        for det in dets:
            shape = self.predictor(image, det)
            shape_vec = []
            for i in xrange(68):
                shape_vec.append(shape.part(i).x)
                shape_vec.append(shape.part(i).y)
            shapes.append(shape_vec)
        # TODO time test
        usetime = (cv2.getTickCount() - start)/cv2.getTickFrequency()
        # print "Use time {}s.".format(usetime)
        return np.array(rectangles), np.array(shapes)
    
    # SAME as calc_landmark_resut but add the pose results
    # Return : 	a numpy array of shape [num_face, 3]
    # Denote of each row:
    #		[pose_x, pose_y, pose_z] all measured in degrees and image's left_top_clockwise is positive pole
    #def calc_full_result(self, image):
    #    rectangles, shapes = self.calc_landmark_result(image)
    #    # parameter rough guess
    #    height, width = image.shape[0:2]
    #    cx = width / 2.0
    #    cy = height / 2.0
    #    fx = width * 500.0 / 640.0
    #    fy = height * 500.0 / 480.0
    #    Z = fx
    #    X = -cx * Z * (1.0/fx)
    #    Y = -cy * Z * (1.0/fy)
    #    camera_matrix = np.array([(fx, 0, cx), (0, fy, cy), (0, 0, 1)])
    #    poses = []
    #    for shape in shapes:
    #        # Correction for orientation
    #        landmarks_2D = shape.reshape(68,2).astype(np.float64)

    #        # Solving the PNP model
    #        vec_trans = np.array([X, Y, Z],dtype=np.float64)
    #        vec_rot = np.array([0,0,0],dtype=np.float64)
    #        retval, vec_rot, vec_trans = cv2.solvePnP(self.landmarks_3D, landmarks_2D, camera_matrix, None, vec_rot, vec_trans, True)
    #        pose_x, pose_y, pose_z = vec_rot[:]
    #        pose_x = math.asin(max(-0.5, min(-pose_x,0.5))/0.5)/3.14*180.0
    #        pose_y = math.asin(max(-0.5, min(pose_y,0.5))/0.5)/3.14*180.0
    #        pose_z = math.asin(max(-0.5, min(pose_z,0.5))/0.5)/3.14*180.0
    #        poses.append([pose_x, pose_y, pose_z])

    #    return rectangles, shapes, poses 

    # Extract two eyes' points from detected shape from calc_landmark_result 
    # Denote of each row:
    #       [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
    def extract_eye_result(self, shapes):
        assert(shapes is not None)
        assert(shapes.shape[0] > 0 and shapes.shape[1] == 136)
        eye_shapes = []
        for shape in shapes:
            eye_shape = []
            eye_shape.append( round((shape[37*2] + shape[38*2] + shape[40*2] + shape[41*2])/4.0) )
            eye_shape.append( round((shape[37*2+1] + shape[38*2+1] + shape[40*2+1] + shape[41*2+1])/4.0) )
            eye_shape.append( round((shape[43*2] + shape[44*2] + shape[46*2] + shape[47*2])/4.0) )
            eye_shape.append( round((shape[43*2+1] + shape[44*2+1] + shape[46*2+1] + shape[47*2+1])/4.0) )     
            eye_shapes.append(eye_shape)
        return np.array(eye_shapes)
    # Extract five face landmark points from detected shape from calc_landmark_result 
    # Denote of each row:
    #       [left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth...]
    def extract_five_result(self, shapes):
        assert(shapes is not None)
        assert(shapes.shape[0] > 0 and shapes.shape[1] == 136)
        eye_shapes = []
        for shape in shapes:
            eye_shape = []
            eye_shape.append( round((shape[37*2] + shape[38*2] + shape[40*2] + shape[41*2])/4.0) )
            eye_shape.append( round((shape[37*2+1] + shape[38*2+1] + shape[40*2+1] + shape[41*2+1])/4.0) )
            eye_shape.append( round((shape[43*2] + shape[44*2] + shape[46*2] + shape[47*2])/4.0) )
            eye_shape.append( round((shape[43*2+1] + shape[44*2+1] + shape[46*2+1] + shape[47*2+1])/4.0) )
            eye_shape.append( round(shape[30*2]) )
            eye_shape.append( round(shape[30*2+1]) )
            eye_shape.append( round(shape[48*2]) )
            eye_shape.append( round(shape[48*2+1]) )
            eye_shape.append( round(shape[54*2]) )
            eye_shape.append( round(shape[54*2+1]) )
            eye_shapes.append(eye_shape)
        return np.array(eye_shapes).astype(np.int)

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
        cv2.imwrite("show.jpg", image)
