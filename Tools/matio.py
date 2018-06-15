import struct
import numpy as np

cv_type_to_dtype = {
	5 : np.dtype('float32'),
	6 : np.dtype('float64')
}

dtype_to_cv_type = {v : k for k,v in cv_type_to_dtype.items()}

def write_mat(f, m):
    """Write mat m to file f"""
    if len(m.shape) == 1:
        rows = m.shape[0]
        cols = 1
    else:
        rows, cols = m.shape
    header = struct.pack('iiii', rows, cols, cols * 4, dtype_to_cv_type[m.dtype])
    f.write(header)
    f.write(m.data)


def read_mat(f):
    """
    Reads an OpenCV mat from the given file opened in binary mode
    """
    rows, cols, stride, type_ = struct.unpack('iiii', f.read(4*4))
    mat = np.fromstring(f.read(rows*stride),dtype=cv_type_to_dtype[type_])
    return mat.reshape(rows,cols)

def load_mat(filename):
    """
    Reads a OpenCV Mat from the given filename
    """
    
    f = open(filename, 'rb')
    mat = read_mat(f)
    f.close()
    return mat

def save_mat(filename, m):
    """Saves mat m to the given filename"""
    f = open(filename, 'wb')
    write_mat(f, m)
    f.close()
    return 

