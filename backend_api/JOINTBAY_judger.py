import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import cPickle as pickle
import os

class JOINTBAY_judger:
    def __init__(self, model_path, dim):
        G_model_path = os.path.join(model_path, "G_model.pkl")
        A_model_path = os.path.join(model_path, "A_model.pkl")
        PCA_model_path = os.path.join(model_path, "pca_model.m")
        if((not os.path.exists(G_model_path)) or (not os.path.exists(A_model_path))):
            raise Exception("{} or {} do not exist.".format(G_model_path, A_model_path))
        if( not os.path.exists(PCA_model_path) ):
	    raise Exception("{} do not exist.".format(PCA_model_path))
        with open(G_model_path, "rb") as Gf:
            self.G = pickle.load(Gf)
        with open(A_model_path, "rb") as Af:
            self.A = pickle.load(Af)
        self.pca = joblib.load(PCA_model_path)
        self.dim = dim
    
    def get_result(self, feature1, feature2):
        assert(np.prod(feature1.shape) == self.dim and np.prod(feature2.shape) == self.dim)
        feats = np.concatenate((feature1.reshape((1, self.dim)), feature2.reshape((1, self.dim))), axis=0)
        base_feats = self.pca.transform(feats)
        base_feat1 = base_feats[0,:]
        base_feat2 = base_feats[1,:]
        return np.dot(np.dot(base_feat1,self.A),np.transpose(base_feat1)) + np.dot(np.dot(base_feat2,self.A),np.transpose(base_feat2)) - 2*np.dot(np.dot(base_feat1,self.G),np.transpose(base_feat2))
