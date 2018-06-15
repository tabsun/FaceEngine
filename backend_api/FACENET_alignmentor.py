import math
import numpy as np
import cv2
# Alignmentor for facenet
class FACENET_alignmentor:
    def __init__(self):
	return

    def _rotate(self, image, angle, center_x, center_y):
        height, width = image.shape[0:2]
        rotateMat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        rotateImg = cv2.warpAffine(image, rotateMat, (2*width, 2*height))

        return rotateImg
    # this process just make the image fit to facenet but no alignment
    # This is because some image may be not able to detect eye points
    # - image : [numpy array] image to process
    # - img : [numpy array] return aligned image
    def get_direct_result(self, image):
        assert(image is not None)
        assert(image.shape[0] > 0 and image.shape[1] > 0)

        width = image.shape[1]
        height = image.shape[0]
        square_w = max(width, height)
        full_image = np.zeros((square_w, square_w, 3), dtype=np.uint8)
        if len(image.shape) == 2 or image.shape[2] == 1:
            full_image[(square_w-height)/2:(square_w-height)/2+height, \
                (square_w-width)/2:(square_w-width)/2+width,:] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            full_image[(square_w-height)/2:(square_w-height)/2+height, \
                (square_w-width)/2:(square_w-width)/2+width,:] = image
        return cv2.resize(full_image, (128,128))

    # - image : [numpy array] image to process [MUST BE BGR SEQUENCE!!!!]
    # - eye_points: [numpy array 1x4 or 4x1] the two eyes' positions [lx,ly,rx,ry]
    # - img : [numpy array] return aligned image
    def get_result(self, image, eye_points):
        assert(image is not None and eye_points is not None)
        assert(image.shape[0] > 0 and image.shape[1] > 0 and \
        set(eye_points.shape) == set([4]))
        
        height = image.shape[0]
        width = image.shape[1]
        lx,ly,rx,ry = eye_points[:]
        scale = math.sqrt( (rx-lx)*(rx-lx) + (ry-ly)*(ry-ly) ) /30.0
        offset_x = int(math.ceil(max(0, 49.0*scale - lx - width)))
        offset_y = int(math.ceil(max(0, 70.0*scale - ly - height)))
        full_image = np.zeros((3*height+2*offset_y, 3*width+2*offset_x, 3), dtype=np.uint8)
        if len(image.shape) == 2 or image.shape[2] == 1:
            full_image[offset_y+height:offset_y+2*height, offset_x+width:offset_x+2*width,:] = \
                cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            full_image[offset_y+height:offset_y+2*height, offset_x+width:offset_x+2*width,:] = \
                image
        
        angle = math.atan( -(ry-ly)/(rx-lx) )*180.0/math.pi if rx!=lx else 0
        lx += width + offset_x
        ly += height + offset_y
        rotate_image = self._rotate(full_image, -angle, lx, ly)
        x = int(round(lx - 49.0*scale))
        y = int(round(ly - 70.0*scale))
        w = int(round(128.0*scale))
        h = w
        align_image = np.array(rotate_image[y:y+h, x:x+w,:], copy=True)
        align_image = cv2.resize(align_image, (128,128))

        return align_image

    # - image : [numpy array] image to process [MUST BE BGR SEQUENCE!!!!]
    # - eye_points: [numpy array 1x4 or 4x1] the two eyes' positions [lx,ly,rx,ry]
    # - img : [numpy array] return aligned image
    def get_result_by_params(self, image, eye_points, dist, anchor_x, anchor_y, dst_width):
        assert(image is not None and eye_points is not None)
        assert(image.shape[0] > 0 and image.shape[1] > 0 and \
        set(eye_points.shape) == set([4]))
        
        height = image.shape[0]
        width = image.shape[1]
        lx,ly,rx,ry = eye_points[:]
        scale = math.sqrt( (rx-lx)*(rx-lx) + (ry-ly)*(ry-ly) ) /dist
        offset_x = int(math.ceil(max(0, anchor_x*scale - lx - width)))
        offset_y = int(math.ceil(max(0, anchor_y*scale - ly - height)))
        full_image = np.zeros((3*height+2*offset_y, 3*width+2*offset_x, 3), dtype=np.uint8)
        if len(image.shape) == 2 or image.shape[2] == 1:
            full_image[offset_y+height:offset_y+2*height, offset_x+width:offset_x+2*width,:] = \
                cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            full_image[offset_y+height:offset_y+2*height, offset_x+width:offset_x+2*width,:] = \
                image
        
        angle = math.atan( -(ry-ly)/(rx-lx) )*180.0/math.pi if rx!=lx else 0
        lx += width + offset_x
        ly += height + offset_y
        rotate_image = self._rotate(full_image, -angle, lx, ly)
        x = int(round(lx - anchor_x*scale))
        y = int(round(ly - anchor_y*scale))
        w = int(round(dst_width*scale))
        h = w
        align_image = np.array(rotate_image[y:y+h, x:x+w,:], copy=True)
        align_image = cv2.resize(align_image, (dst_width,dst_width))

        return align_image
