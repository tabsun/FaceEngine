# -*- coding: utf-8 -*- 
import os, cv2, base64, re
import tensorflow as tf
import numpy as np

# FACE ENGINE
import sys
sys.path.insert(0, '../backend_api')
from matio import load_mat, save_mat

CELEPOOL_FOLDER = '../celefeature_files'
#CELEPOOL_FOLDER = '../celefeature_files'

def get_score(feature1, feature2):
    assert( feature1.size == feature2.size )
    s = np.dot(feature1.reshape(1,-1), feature2.reshape(-1,1))
    s = s[0,0]
    if( s<0.5 ):
        s = (s + 1.0)/3.0
    return s

def inner_recognize(file_name, feature):
    global cele_features
    global cele_names
    global cele_file_names

    scores = []
    names = []
    for i in xrange(len(cele_features)):
	if(file_name == cele_file_names[i]):
	    continue
        cele_feat = cele_features[i]
	cele_name = cele_names[i]
	score = get_score(feature, cele_feat)
	if(cele_name in names):
	    id = names.index(cele_name)
	    if(scores[id] < score):
	        scores[id] = score
	    continue

	if(len(scores) < 10):
            scores.append(score)
	    names.append(cele_name)
	else:
	    min_value = min(scores)
	    min_index = scores.index(min_value)
	    if(score > min_value):
		scores[min_index] = score
		names[min_index] = cele_name
    index = sorted(range(len(scores)), key=lambda k:scores[k], reverse=True)
    sort_scores = [scores[id] for id in index]
    sort_names = [names[id] for id in index]
    return sort_names, sort_scores

def load_cele_features():
    features = []
    names = []
    file_names = []
    celefeature_folder = CELEPOOL_FOLDER
    for the_file in os.listdir(celefeature_folder):
	name = the_file.rsplit('#', 1)[0]
        file_name = the_file.rsplit('.')[0]
	file_path = os.path.join(celefeature_folder, the_file)
	feature = load_mat(file_path)
        features.append(feature)
	names.append(name)
	file_names.append(file_name)
    return features, names, file_names

if __name__=='__main__':
    # Load all celebrity features
    cele_features, cele_names, cele_file_names = load_cele_features()
    print "Load Done"

    top_rates = [0] * 10

    ids = []
    for i in xrange(len(cele_features)):
	cele_name = cele_names[i]
	if(cele_names.count(cele_name) > 1):
	    ids.append(i)
    count = 0
    f = open("missed.txt","w")
    for i in ids:
	print "{} in {}".format(count, len(ids))
	count += 1
	cele_feature = cele_features[i]
	cele_name = cele_names[i]
	cele_file_name = cele_file_names[i]
	sort_names, _ = inner_recognize(cele_file_name, cele_feature)
        if(cele_name not in sort_names):
	    f.write(cele_file_name)
	    f.write("\n")
        for top_id in xrange(1,11):
	    if(cele_name in sort_names[0:top_id]):
		top_rates[top_id-1] += 1
    top_score = ["%.2f" % (elem*100.0/len(ids)) for elem in top_rates]
    f.close()
    print top_score
