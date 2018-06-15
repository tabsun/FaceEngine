import numpy as np
import numpy.linalg as LA
import cv2, os
import math

def Euler2RotationMatrix(eulerAngles):
	s1, s2, s3 = np.sin(eulerAngles)[:]
	c1, c2, c3 = np.cos(eulerAngles)[:]
	rotation_matrix = np.array([c2 * c3, -c2 *s3, s2, \
    c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1, \
	s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]).reshape((3,3))
	return rotation_matrix

def RotationMatrix2Euler(rotation_matrix):
    q0 = math.sqrt( 1 + rotation_matrix[0,0] + rotation_matrix[1,1] + rotation_matrix[2,2] ) / 2.0
    q1 = (rotation_matrix[2,1] - rotation_matrix[1,2]) / (4.0*q0) 
    q2 = (rotation_matrix[0,2] - rotation_matrix[2,0]) / (4.0*q0) 
    q3 = (rotation_matrix[1,0] - rotation_matrix[0,1]) / (4.0*q0) 
    t1 = 2.0 * (q0*q2 + q1*q3)

    yaw  = math.asin(2.0 * (q0*q2 + q1*q3))
    pitch= math.atan2(2.0 * (q0*q1-q2*q3), q0*q0-q1*q1-q2*q2+q3*q3)
    roll = math.atan2(2.0 * (q0*q3-q1*q2), q0*q0+q1*q1-q2*q2-q3*q3)
    
    return np.array([pitch, yaw, roll])

def AxisAngle2RotationMatrix(axis_angle):
    dst, jacob = cv2.Rodrigues(axis_angle)
    return dst

def RotationMatrix2AxisAngle(rotation_matrix):
    dst, jacob = cv2.Rodrigues(rotation_matrix)
    return dst

def AxisAngle2Euler(axis_angle):
    rotation_matrix,_ = cv2.Rodrigues(axis_angle)
    return RotationMatrix2Euler(rotation_matrix)

def CalcShape2D(mean_shape, princ_comp, params_local,  params_global):
    n = mean_shape.shape[0] / 3
    s = params_global[0] # scaling factor
    tx = params_global[4] # x offset
    ty = params_global[5] # y offset
    euler = np.array( [params_global[1],params_global[2],params_global[3]] )
    currRot = Euler2RotationMatrix(euler)	
    # get the 3D shape of the object
    Shape_3D = mean_shape + np.matmul(princ_comp,params_local)
    # create the 2D shape matrix (if it has not been defined yet)
    out_shape = np.zeros((n*2, 1), dtype=np.float64)
    # for every vertex
    for i in xrange(n):
	    # Transform this using the weak-perspective mapping to 2D from 3D
        out_shape[i] = s * ( currRot[0,0] * Shape_3D[i] + currRot[0,1] * Shape_3D[i+n] + currRot[0,2] * Shape_3D[i+n*2] ) + tx
        out_shape[i+n] = s * ( currRot[1,0] * Shape_3D[i] + currRot[1,1] * Shape_3D[i+n] + currRot[1,2] * Shape_3D[i+n*2] ) + ty
    
    return out_shape

def CalcBoundingBox(mean_shape, princ_comp, params_local, params_global):
    n = mean_shape.shape[0]/3
    curr_shape = CalcShape2D(mean_shape, princ_comp, params_local, params_global)
    min_x, min_y = np.amin(curr_shape.reshape((n,2),order='F'), axis=0)
    max_x, max_y = np.amax(curr_shape.reshape((n,2),order='F'), axis=0)
    return np.array([int(min_x), int(min_y), int(abs(max_x-min_x)), int(abs(max_y-min_y))])

def CalcShape3D(mean_shape, princ_comp, params_local):
    return mean_shape + np.matmul(princ_comp, params_local)

def ComputeJacobian(mean_shape, princ_comp, params_local, params_global):
    n = mean_shape.shape[0] / 3
    m = princ_comp.shape[1]

    s = params_global[0]
    shape_3D = CalcShape3D(mean_shape, princ_comp, params_local)
    euler = np.array([params_global[1], params_global[2], params_global[3]],dtype=np.float)
    currRot = Euler2RotationMatrix(euler)

    r11,r12,r13 = currRot[0,:]
    r21,r22,r23 = currRot[1,:]
    r31,r32,r33 = currRot[2,:]
    Jacobian = np.zeros( (n * 2, 6 + m), dtype=float )
    for i in xrange(n):
        X = shape_3D[i, 0]
        Y = shape_3D[i+n, 0]
        Z = shape_3D[i+2*n, 0]
        
        Jacobian[i,0] = (X  * r11 + Y * r12 + Z * r13)
        Jacobian[i+n,0] = (X  * r21 + Y * r22 + Z * r23)
        Jacobian[i,1] = (s * (Y * r13 - Z * r12) )
        Jacobian[i+n,1] = (s * (Y * r23 - Z * r22) )
        Jacobian[i, 2] = (-s * (X * r13 - Z * r11))
        Jacobian[i+n,2] = (-s * (X * r23 - Z * r21))
        Jacobian[i, 3] = (s * (X * r12 - Y * r11) )
        Jacobian[i+n, 3] = (s * (X * r22 - Y * r21) )
        Jacobian[i, 4] = 1.0
        Jacobian[i+n,4] = 0.0
        Jacobian[i, 5] = 0.0
        Jacobian[i+n, 5] = 1.0

        Jacobian[i, 6:] = s*(r11*princ_comp[i,:] + r12*princ_comp[i+n,:] + r13*princ_comp[i+2*n,:])
        Jacobian[i+n, 6:] = s*(r21*princ_comp[i,:] + r22*princ_comp[i+n,:] + r23*princ_comp[i+2*n,:])

    return Jacobian, np.transpose(Jacobian)
def Orthonormalise(R):
    w, u, vt = cv2.SVDecomp(R, flags=cv2.SVD_MODIFY_A)
	# get the orthogonal matrix from the initial rotation matrix
    X = u*vt
	# This makes sure that the handedness is preserved and no reflection happened
	# by making sure the determinant is 1 and not -1
    W = np.eye(3)
    d = LA.det(X)
    W[2,2] = d
    Rt = u * W * vt

    return Rt

def UpdateModelParameters(delta_p, params_local, params_global):
    params_global[0] += delta_p[0,0]
    params_global[4] += delta_p[4,0]
    params_global[5] += delta_p[5,0]
    eulerGlobal = np.array([params_global[1], params_global[2], params_global[3]])
    R1 = Euler2RotationMatrix(eulerGlobal)
    R2 = np.eye(3)

    R2[2,1] = delta_p[1,0]
    R2[0,2] = delta_p[2,0]
    R2[1,0] = delta_p[3,0]
    R2[1,2] = -1.0*R2[2,1]
    R2[2,0] = -1.0*R2[0,2]
    R2[0,1] = -1.0*R2[1,0]
    Orthonormalise(R2)

    R3 = np.matmul(R1, R2)
    axis_angle = RotationMatrix2AxisAngle(R3)
    euler = AxisAngle2Euler(axis_angle)
    params_global[1] = euler[0]
    params_global[2] = euler[1]
    params_global[3] = euler[2]
    if(delta_p.shape[0] > 6):
        params_local = params_local + delta_p[6:, :]

    return params_global, params_local
    
# landmark_location is a 68 x 2 Mat denotes the detected_landmarks_2D, if one point cannot be seen then it's set to 0
def CalcParams(landmark_locations, mean_shape, princ_comp, eigen_values):
    m = princ_comp.shape[1]

    # read mean_shape_3D and princ_comp_3D
    vis_ind = (landmark_locations[0:68,0] != 0)
    vis_ind = np.concatenate([vis_ind,vis_ind,vis_ind])
    mean_shape = mean_shape[vis_ind, :]
    princ_comp = princ_comp[vis_ind, :]
    
    n = mean_shape.shape[0] / 3
    landmark_locs_vis = landmark_locations[landmark_locations[:,0] != 0, :]
    
    min_x, min_y = np.amin(landmark_locs_vis, axis=0)
    max_x, max_y = np.amax(landmark_locs_vis, axis=0)
    landmark_locs_vis = landmark_locs_vis.reshape((n*2,1),order='F')

    model_bbox = CalcBoundingBox(mean_shape, princ_comp, np.zeros((m,1), dtype=np.float), np.array([1,0,0,0,0,0], dtype=np.float))
    bbox = np.array([int(min_x), int(min_y), int(abs(max_x-min_x)), int(abs(max_y-min_y))])
    scaling = ((bbox[2] * 1.0 / model_bbox[2]) + (bbox[3]* 1.0 / model_bbox[3])) / 2.0

    rotation_init = np.zeros((3,1), dtype=float)
    # prepare the loop needed parameters
    R = Euler2RotationMatrix(rotation_init)
    translation = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float)
    loc_params = np.zeros((princ_comp.shape[1],1), dtype=np.float)
    glob_params = np.array([scaling, rotation_init[0], rotation_init[1], rotation_init[2], translation[0], translation[1]])
    shape_3D = mean_shape + np.matmul(princ_comp, loc_params)

    curr_shape = np.zeros((2*n,1), dtype=np.float)
    for i in xrange(n):
		# Transform this using the weak-perspective mapping to 2D from 3D
        curr_shape[i  ,0] = scaling * ( R[0,0] * shape_3D[i, 0] + R[0,1] * shape_3D[i+n  ,0] + R[0,2] * shape_3D[i+n*2,0] ) + translation[0]
        curr_shape[i+n,0] = scaling * ( R[1,0] * shape_3D[i, 0] + R[1,1] * shape_3D[i+n  ,0] + R[1,2] * shape_3D[i+n*2,0] ) + translation[1]

    currError = LA.norm(curr_shape - landmark_locs_vis)

    regFactor = 1.0
    regularisations = np.zeros((m+6,1), np.float)
    regularisations[6:,0] = regFactor / eigen_values
    regularisations = np.diagflat(regularisations)
    not_improved_in = 0
    scaling = 0.5723671
    for i in xrange(1000):
        shape_3D = mean_shape + np.matmul(princ_comp, loc_params)
        shape_3D = shape_3D.reshape((3, -1))
        R_2D = np.array([R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]).reshape((2,3))
        curr_shape_2D = scaling * np.matmul(np.transpose(shape_3D), np.transpose(R_2D))
        curr_shape_2D[:,0] = curr_shape_2D[:,0] + translation[0]
        curr_shape_2D[:,1] = curr_shape_2D[:,1] + translation[1]
        curr_shape_2D = np.transpose(curr_shape_2D).reshape((n*2, -1))

        error_resid = landmark_locs_vis - curr_shape_2D
        J, J_w_t = ComputeJacobian(mean_shape, princ_comp, loc_params, glob_params)
        J_w_t_m = np.matmul(J_w_t, error_resid)
        J_w_t_m[6:] = J_w_t_m[6:] - np.matmul(regularisations[6:,6:], loc_params)
        Hessian = np.matmul(J_w_t, J) + regularisations
        # cv::solve
        _, param_update = cv2.solve(Hessian, J_w_t_m, flags=cv2.DECOMP_CHOLESKY)
        param_update = param_update * 0.5
        glob_params, loc_params = UpdateModelParameters(param_update, loc_params, glob_params)
        scaling = glob_params[0]
        rotation_init[0] = glob_params[1]
        rotation_init[1] = glob_params[2]
        rotation_init[2] = glob_params[3]

        translation[0] = glob_params[4]
        translation[1] = glob_params[5]
        
        R = Euler2RotationMatrix(rotation_init)
        R_2D[0:2,0:3] = R[0:2,0:3] 

        curr_shape_2D = scaling * np.matmul(np.transpose(shape_3D), np.transpose(R_2D))
        curr_shape_2D[:,0] = curr_shape_2D[:,0] + translation[0]
        curr_shape_2D[:,1] = curr_shape_2D[:,1] + translation[1]

        curr_shape_2D = np.transpose(curr_shape_2D).reshape(n * 2, -1)
        
        error = LA.norm(curr_shape_2D - landmark_locs_vis)  
        if(0.999 * currError < error):
            not_improved_in = not_improved_in + 1
            if(not_improved_in == 5):
                break
        currError = error
    
    return glob_params, loc_params


class ANGLE_judger:
    def __init__(self, model_path, type='dynamic'):
	if(type == 'dynamic'):
	    eigen_path = os.path.join(model_path, "eigen_value.txt")
	    mean_path = os.path.join(model_path, "mean_shape.txt")
	    princ_path = os.path.join(model_path, "princ_comp.txt")
	    if(not os.path.exists(eigen_path) or not os.path.exists(mean_path) or not os.path.exists(princ_path)):
	        raise Exception( "{} or {} or {} do not exists.".format(eigen_path, mean_path, princ_path))
            # first you need to read in mean_shape / princ_comp / eigen_values
            self.princ_comp = np.loadtxt(princ_path, dtype=np.float).reshape((204,34))
            self.mean_shape = np.loadtxt(mean_path, dtype=np.float).reshape((204,1))
            self.eigen_values = np.loadtxt(eigen_path, dtype=np.float).reshape((1,34))
        self.landmarks_3D = np.array([-75.31452318321439, -38.34471459717126, 49.26396240078913,-75.80202922424679, -16.31303384119447, 53.06084660014774, -73.82597716618611, 6.87459402799943, 55.34202324078976, -69.93707449150007, 29.1533466182599, 53.65072895141991, -61.78025166899538, 48.66180356286463, 45.76265021415519, -49.04016739555473, 64.76047179539211, 32.9317961427254, -33.62663591327581, 75.99609280185886, 14.42724844093341, -15.82291707647611, 84.15895057694331, -2.278597131383455, 3.224108513099121, 86.50308193842028, -2.289249682827585, 22.00897418212334, 84.06167586588964, 7.684029939496918, 38.01236118575019, 74.80029681078426, 25.52886954481255, 50.45743038615099, 61.68586083785743, 38.23285028839483, 58.0389898035469, 45.43298595989082, 45.03626427225792, 61.77747909692723, 26.17436180813345, 43.77597303746869, 63.158783290275, 5.325479567932502, 38.96781785433828, 63.95480488476984, -16.55062433770693, 31.74461906872116, 63.40856749538791, -37.50125902733683, 27.63717726712692, -55.56464504279378, -58.19294884869319, 6.221490796207339, -45.136376771246, -64.70450816166314, 2.61457864155236, -32.85778411410283, -64.73490050515963, 0.3567964690260295, -22.07035060391473, -60.53324719540702, -2.327404092013576, -12.6420994230173, -55.19189169930144, -5.725341358361012, 16.52779458398911, -55.63572426203642, -8.332430170540739, 26.55222419022155, -59.6511565390853, -4.651166929443567, 36.56890085863404, -61.61200161217572, -0.6199366120584231, 46.10439372147976, -60.71649521457847, 4.350831182246043, 52.89836029628062, -53.87916796188836, 9.589029770033134, 2.054525479806006, -41.62762382503091, -7.419967926244057, 2.360432735147678, -28.27391226207461, -15.78866955209314, 2.739551963296541, -14.91334212325783, -23.85913013715855, 3.128209076964043, -1.027642269842878, -31.29205066613887, -11.24962249360924, 4.473252731486111, -17.65164210424473, -4.955327112281717, 7.558797876242555, -21.04004280959795, 1.528314822745628, 10.1846952180661, -23.15875488480389, 7.952515430230305, 8.033705325941458, -21.11887665167271, 13.4194155758756, 5.409808372276553, -17.3808328711752, -41.09568821723335, -41.11571146723735, 1.401355420669464, -33.85695879567623, -44.16637820586832, -0.2284977378446009, -25.53080826252835, -43.80199503588361, -1.09117799800995, -18.67278039524081, -39.99617897579242, -0.9367286155310053, -26.37433533120626, -38.04941363038587, -2.503430365949899, -34.60417830733062, -38.24475398787638, -1.938708819581009, 17.13470268612971, -38.3409860149955, 3.044301738607214, 25.07752945783057, -41.57182897885824, 2.497524599507898, 32.90289473580152, -41.20212989315934, 3.472830468518154, 38.77913353075716, -37.88960924409194, 5.999498942851558, 32.92172816041038, -35.85801410291159, 2.91188860114035, 25.34350119829321, -36.28139720170071, 1.372623236802369, -22.33390705744408, 26.12710420724033, -10.52548107151055, -12.57913377950485, 21.36269231763391, -20.4426450317165, -4.373818868095719, 20.34823771134812, -21.8331858363356, 1.165691606218703, 22.04923641764354, -21.89821461286107, 7.261938115926549, 20.76243977000641, -19.84835858280928, 14.00321423236733, 22.83080270021395, -15.78996035711305, 21.09752184736759, 26.75871631183023, -7.999944092822259, 13.54758182834182, 32.36397985618099, -20.58614414255182, 6.916016546352237, 34.09477431038195, -24.82610091700534, 0.425344498506479, 34.49017822945682, -27.0838616939405, -5.359393273114829, 33.81619369535882, -27.06107299768875, -13.07923516195734, 31.5166228484679, -24.90837633398865, -18.41326230535908, 25.9006801602609, -13.25221970027098, -4.522567451095701, 26.09937338968868, -21.14414502255659, 0.9804318169366493, 26.58093361662769, -21.45171768870722, 7.174253852247737, 26.09627336656524, -18.61956228805804, 16.91713764798056, 26.79920358705355, -12.13417672766182, 6.938373140734785, 25.92281289328823, -20.105804574834, 0.7183685990515277, 26.60940549839257, -22.98519844701312, -4.75940270633021, 26.14362354101845, -22.75068083737279], dtype=np.float64).reshape((68,3)) 

    # shape - numpy array with size = 136 x 1, denoting 68 face landmarks [x0, x1, x2, ..., y0, y1, y2...].T
    # width - image width
    # height - image height corresponding to shape's positions
    def CalcPose(self, shape, width, height, type='dynamic'):
        assert(np.prod(shape.shape) == 136)
	shape = shape.reshape((136,1))
	min_x, min_y = np.amin(shape.reshape((68,2)), axis=0)
	max_x, max_y = np.amax(shape.reshape((68,2)), axis=0)
	#width = int(round(1.5*(max_x - min_x)))
	#height = int(round(1.5*(max_y - min_y)))
        # parameter rough guess
        cx = width / 2.0
        cy = height / 2.0
        fx = width * 500.0 / 640.0
        fy = height * 500.0 / 480.0
        Z = fx
        X = -cx * Z * (1.0/fx)
        Y = -cy * Z * (1.0/fy)
        camera_matrix = np.array([(fx, 0, cx), (0, fy, cy), (0, 0, 1)])
        # Correction for orientation
        landmarks_2D = shape.reshape((68,2)).astype(np.float64)
	if(type == 'dynamic'):
            params_global, params_local = CalcParams(landmarks_2D, self.mean_shape, self.princ_comp, self.eigen_values)
            landmarks_3D = CalcShape3D(self.mean_shape, self.princ_comp, params_local).reshape((-1,3),order='F').astype(np.float64)
	else:
	    landmarks_3D = self.landmarks_3D
        # Solving the PNP model
        vec_trans = np.array([X, Y, Z],dtype=np.float64)
        vec_rot = np.array([0,0,0],dtype=np.float64)
        retval, vec_rot, vec_trans = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix, np.zeros((4,1),dtype=np.float64), rvec=vec_rot, tvec=vec_trans, useExtrinsicGuess=1, flags=0) #cv2.SOLVEPNP_ITERATIVE)
        
        euler = AxisAngle2Euler(vec_rot)
        pose_x, pose_y, pose_z = vec_rot[:]
        pose_x = max(-0.5, min(-pose_x,0.5))/0.5*60.0 # recorrect base = 0.12
        pose_y = max(-0.5, min(pose_y,0.5))/0.5*60.0
        pose_z = max(-0.5, min(pose_z,0.5))/0.5*60.0
        pose = [pose_x, pose_y, pose_z]

        return pose

