# -*- coding: utf-8 -*- 
import os, cv2, base64, re, json, time, md5, skvideo.io, math, threading, requests
import tensorflow as tf
import numpy as np
import shutil
from urllib2 import urlopen
from werkzeug.utils import secure_filename
from flask import Flask, request, Response, redirect, url_for, make_response, send_from_directory, render_template, jsonify



# FACE ENGINE
import sys
sys.path.insert(0, './backend_api')
from FACE_engine import FACE_engine
from matio import load_mat, save_mat

UPLOAD_FOLDER = './uploaded_images'
FEATURE_FOLDER = './feature_files'
CELEPOOL_FOLDER = './celefeature_files'
BLACKPOOL_FOLDER = './black_celefeature_files'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'JPG', 'jpeg' ])
ALLOWED_VIDEO_EXTENSIONS = set(['avi', 'mp4'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FEATURE_FOLDER'] = FEATURE_FOLDER
app.config['CELEPOOL_FOLDER'] = CELEPOOL_FOLDER
app.config['BLACKPOOL_FOLDER'] = BLACKPOOL_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def log_info(text):
    with open("log.txt", "w") as f:
	f.write(text)
    return

def process(file):
    log_info(file.filename)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
	# read into numpy array and get ready to be processed
	image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # save this image and delete the oldest one if the upload_folder if too full
	upload_dir = app.config['UPLOAD_FOLDER']
        saved_file_names = [name for name in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir,name))]
	modify_times = [ os.path.getmtime(os.path.join(upload_dir, name)) for name in saved_file_names ]
	if(len(saved_file_names) > 10):
	    delete_file_name = saved_file_names[ modify_times.index(min(modify_times)) ]
            os.remove( os.path.join( upload_dir, delete_file_name ) )
        file.save( os.path.join( upload_dir, filename ) )

        
	# return the image to display
        retval, image_buffer = cv2.imencode('.png', image)
	png_as_text = base64.b64encode(image_buffer)
	return png_as_text
    else:
	return None

def serialize(id, rect, shape, pose):
    return { 'face_id': id,
	     'rect': rect.tolist(),
	     'shape': shape.tolist(),
	     'pose': pose.tolist(),
	     }

def serialize2(id, rect, shape, pose, attribute, name):
    attribute_str = ["", "", "", ""]
    # Race
    if(abs(attribute[0]-0.0) < 0.3):
	attribute_str[0] = "亚裔"
    if(abs(attribute[0]-1.0) < 0.3):
	attribute_str[0] = "白人"
    if(abs(attribute[0]-2.0) < 0.3):
	attribute_str[0] = "非裔"
    if(abs(attribute[0]-3.0) < 0.3):
        attribute_str[0] = "其他"
        
    # Gender
    if(attribute[1]==0):
	attribute_str[1] = "女"
    else:
	attribute_str[1] = "男"
    # Smile
    if(abs(attribute[2]-0.0) < 0.3):
	attribute_str[2] = "平静"
    if(abs(attribute[2]-1.0) < 0.3):
	attribute_str[2] = "微笑"
    if(abs(attribute[2]-2.0) < 0.3):
	attribute_str[2] = "大笑"
    # Age
    attribute_str[3] = "%.0f" % attribute[3]
    return { 'face_id': id,
	     'rect': rect.tolist(),
	     'shape': shape.tolist(),
	     'pose': pose.tolist(),
	     'attribute': attribute_str,
	     'name': name ,}

def serialize3(name, score):
    return { 'name': name,
	     'similarity': score,
	     }

@app.route('/api/calc_rect_shape_pose_by_url', methods=['POST'])
def calc_part_result_by_url():
    global face_id
    global max_face_id_num
    if request.method == 'POST':
        # verify the keys
	ACCESS_KEY = request.form.get('ACCESS_KEY').encode("utf-8")
	TIMESTAMP = request.form.get('TIMESTAMP').encode("utf-8")
	SIGN_KEY = request.form.get('SIGN_KEY').encode("utf-8")
	user_id = request.form.get('user_id').encode("utf-8")
        image_url = request.form.get('image_url').encode("utf-8")
	if ACCESS_KEY and TIMESTAMP and SIGN_KEY and user_id >= 0 and image_url:
            check_result, message = _check_security(ACCESS_KEY,TIMESTAMP,SIGN_KEY,user_id)
	    if not check_result:
		return message
	else:
	    return error_resp(2, "LACK_PARAM ERROR")
		
    	try:
            req = urlopen(image_url)
            img_array = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
	except:
	    return error_resp(3, "GET_IMAGE ERROR")
	if image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
            return error_resp(2, "IMAGE_DATA ERROR")

        # Begin to process
        rects, shapes, poses = eng.calc_full_result(image)

        # Calc features and store them
	features = eng.calc_feature_result(image, shapes)

        # Do not need the adjust for 106 landmarks
        # Calc 106 shape
	# shapes = eng.calc_106_landmarks(image, shapes)

        results = []
	for i in xrange(len(rects)):
            rect = rects[i,:]
	    shape = shapes[i,:]
	    pose = poses[i,:]
	    feature = features[i,:]
	    results.append( serialize(face_id, rect, shape, pose))
	    # Save to FEATURE_FOLDER
	    save_path = os.path.join(app.config['FEATURE_FOLDER'], "{}.bin".format(face_id))
            save_mat(save_path, feature)
	    face_id = (face_id+1) % max_face_id_num
	    
	resp = jsonify(status=1,length=len(results), result=results)
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp
    else:
	return error_resp(2, "METHOD ERROR")

@app.route('/api/calc_full_result_by_url', methods=['POST'])
def calc_full_result_by_url():
    global face_id
    global max_face_id_num
    if request.method == 'POST':
        # verify the keys
	ACCESS_KEY = request.form.get('ACCESS_KEY').encode("utf-8")
	TIMESTAMP = request.form.get('TIMESTAMP').encode("utf-8")
	SIGN_KEY = request.form.get('SIGN_KEY').encode("utf-8")
	user_id = request.form.get('user_id').encode("utf-8")
        image_url = request.form.get('image_url').encode("utf-8")
	if ACCESS_KEY and TIMESTAMP and SIGN_KEY and user_id >= 0 and image_url:
            check_result, message = _check_security(ACCESS_KEY,TIMESTAMP,SIGN_KEY,user_id)
	    if not check_result:
		return message
	else:
	    return error_resp(2, "LACK_PARAM ERROR")
		
    	try:
            req = urlopen(image_url)
            img_array = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
	except:
	    return error_resp(3, "GET_IMAGE ERROR")
        	
	if image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
            return error_resp(2, "IMAGE_DATA ERROR")

        # Begin to process
        rects, shapes, poses = eng.calc_full_result(image)

        # Calc features and store them
        id_features = eng.calc_feature_result(image, shapes)
        
        # Calc attributes data
        attributes = eng.calc_attribute_result(image, shapes)

        # Calc 106 shape
        if(len(rects) > 0):
	    shapes = eng.calc_106_landmarks(image, shapes)
        results = []
	for i in xrange(len(rects)):
            rect = rects[i,:]
	    shape = shapes[i,:]
	    pose = poses[i,:]
            attribute = attributes[i,:].tolist()

	    id_feature = id_features[i,:]
	    names, scores = inner_recognize(id_feature)
	    name = names[ scores.index(max(scores)) ]
	    results.append( serialize2(face_id, rect, shape, pose, attribute, name))
	    # Save to FEATURE_FOLDER
	    save_path = os.path.join(app.config['FEATURE_FOLDER'], "{}.bin".format(face_id))
            save_mat(save_path, id_feature)
	    face_id = (face_id+1) % max_face_id_num
	    
	resp = jsonify(status=1,length=len(results), result=results)
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp
    else:
	return error_resp(2, "METHOD ERROR")

def error_resp(status_id,message_str):
    resp = jsonify(status=status_id, error_message=message_str)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def _sign(secret_key, params=None):
    if params:
        params = sorted(params)
        sign_string = secret_key
        for i in params:
            key, value = i
 	    sign_string += str(key)
 	    sign_string += str(value)
 	sign_string += secret_key
 	hash_string = md5.new()
 	hash_string.update(sign_string)
 	return hash_string.hexdigest().upper()
    else:
       return ''

def _check_security(ACCESS_KEY,TIMESTAMP,SIGN_KEY,user_id):
    global acc_key
    global sec_key
    # time verify
    local_timestamp = int(time.time()) - time_error 
    try:
        get_timestamp = int(TIMESTAMP)
    except:
        return False, error_resp(2, "TIMESTAMP ERROR")
    if abs(int(TIMESTAMP)-local_timestamp) > 600:
        return False, error_resp(2, "TIMESTAMP ERROR")
    # user_id verify
    try:
        user_id_num = int(user_id)
    except:
	return False, error_resp(2, "user_id ERROR")
    if user_id_num is not 1:
        return False, error_resp(2, "user_id ERROR")
    # 
    params = [("ACCESS_KEY", ACCESS_KEY),("TIMESTAMP", TIMESTAMP),("user_id", int(user_id))]
    if ACCESS_KEY != acc_key:
        return False, error_resp(2, "ACCESS_KEY ERROR")
    if SIGN_KEY != _sign(sec_key, params):
        return False, error_resp(2, "SIGN_KEY ERROR")
    return True, ''

@app.route('/api/calc_rect_shape_pose_by_image', methods=['POST'])
def calc_part_result():
    global face_id
    global max_face_id_num
    if request.method == 'POST':
	# verify the keys
	ACCESS_KEY = request.form.get('ACCESS_KEY').encode("utf-8")
	TIMESTAMP = request.form.get('TIMESTAMP').encode("utf-8")
	SIGN_KEY = request.form.get('SIGN_KEY').encode("utf-8")
	user_id = request.form.get('user_id').encode("utf-8")
	image = request.form.get('image').encode("utf-8")
	if ACCESS_KEY and TIMESTAMP and SIGN_KEY and user_id >= 0 and image:
            check_result, message = _check_security(ACCESS_KEY,TIMESTAMP,SIGN_KEY,user_id)
	    if not check_result:
		return message
	else:
	    return error_resp(2, "LACK_PARAM ERROR")

        imgstr = re.search(r'data:image/jpeg;base64,(.*)', image)
	#imgstr = re.search(r'data:image/jpeg;base64,(.*)', request.data)
	if(imgstr is None):
	    return error_resp(2, "IMAGE_DATA ERROR")
        if(not imgstr):
	    return error_resp(2, "IMAGE_DATA ERROR")
        imgstr = imgstr.group(1)
	imgstr = imgstr.decode("base64")
	try:
	    image = cv2.imdecode(np.fromstring(imgstr, np.int8), cv2.IMREAD_COLOR)
	    if(image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0):
                return error_resp(2, "IMAGE_DATA ERROR")
	except:
            return error_resp(2, "IMAGE_DATA ERROR")
        # Begin to process
        rects, shapes, poses = eng.calc_full_result(image)

        # Calc features and store them
	features = eng.calc_feature_result(image, shapes)

        # Do not need the adjust for 106 landmarks
        # Calc 106 shape
	# shapes = eng.calc_106_landmarks(image, shapes)

        results = []
	for i in xrange(len(rects)):
            rect = rects[i,:]
	    shape = shapes[i,:]
	    pose = poses[i,:]
	    feature = features[i,:]
	    results.append( serialize(face_id, rect, shape, pose))
	    # Save to FEATURE_FOLDER
	    save_path = os.path.join(app.config['FEATURE_FOLDER'], "{}.bin".format(face_id))
            save_mat(save_path, feature)
	    face_id = (face_id+1) % max_face_id_num
	    
	resp = jsonify(status=1, length=len(results), result=results)
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp
    else:
	return error_resp(2, "METHOD ERROR")

@app.route('/api/calc_full_result_by_image', methods=['POST'])
def calc_full_result():
    global face_id
    global max_face_id_num
    if request.method == 'POST':
        # verify the keys
	ACCESS_KEY = request.form.get('ACCESS_KEY').encode("utf-8")
	TIMESTAMP = request.form.get('TIMESTAMP').encode("utf-8")
	SIGN_KEY = request.form.get('SIGN_KEY').encode("utf-8")
	user_id = request.form.get('user_id').encode("utf-8")
	image = request.form.get('image').encode("utf-8")
	if ACCESS_KEY and TIMESTAMP and SIGN_KEY and user_id >= 0 and image:
            check_result, message = _check_security(ACCESS_KEY,TIMESTAMP,SIGN_KEY,user_id)
	    if not check_result:
		return message
	else:
	    return error_resp(2, "LACK_PARAM ERROR")
		
	imgstr = re.search(r'image/jpeg;base64,(.*)', image)
	#imgstr = re.search(r'data:image/jpeg;base64,(.*)', request.data)
	if(imgstr is None):
	    return error_resp(2, "IMAGE_DATA ERROR")
        if(not imgstr):
	    return error_resp(2, "IMAGE_DATA ERROR")
        imgstr = imgstr.group(1)
	imgstr = imgstr.decode("base64")
	try:
	    image = cv2.imdecode(np.fromstring(imgstr, np.int8), cv2.IMREAD_COLOR)
	    if(image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0):
                return error_resp(2, "IMAGE_DATA ERROR")
	except:
            return error_resp(2, "IMAGE_DATA ERROR")

        # Begin to process
        rects, shapes, poses = eng.calc_full_result(image)

        # Calc features and store them
        id_features = eng.calc_feature_result(image, shapes)

        # Calc attributes data
        attributes = eng.calc_attribute_result(image, shapes)

        # Calc 106 shape
	shapes = eng.calc_106_landmarks(image, shapes)

        results = []
	for i in xrange(len(rects)):
            rect = rects[i,:]
	    shape = shapes[i,:]
	    pose = poses[i,:]
            attribute = attributes[i,:].tolist()

	    id_feature = id_features[i,:]
	    names, scores = inner_recognize(id_feature)
	    name = names[ scores.index(max(scores)) ]
	    results.append( serialize2(face_id, rect, shape, pose, attribute, name))
	    # Save to FEATURE_FOLDER
	    save_path = os.path.join(app.config['FEATURE_FOLDER'], "{}.bin".format(face_id))
            save_mat(save_path, id_feature)
	    face_id = (face_id+1) % max_face_id_num
	    
	resp = jsonify(status=1, length=len(results), result=results)
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp
    else:
	return "Not proper method"

@app.route('/api/compare', methods=['POST'])
def compare():
    # verify the keys
    ACCESS_KEY = request.form.get('ACCESS_KEY').encode("utf-8")
    TIMESTAMP = request.form.get('TIMESTAMP').encode("utf-8")
    SIGN_KEY = request.form.get('SIGN_KEY').encode("utf-8")
    user_id = request.form.get('user_id').encode("utf-8")
    id_0 = request.form.get('id_0').encode("utf-8")
    id_1 = request.form.get('id_1').encode("utf-8")
    if ACCESS_KEY and TIMESTAMP and SIGN_KEY and user_id >= 0 and id_0 >= 0 and id_1 >= 0:
        check_result, message = _check_security(ACCESS_KEY,TIMESTAMP,SIGN_KEY,user_id)
        if not check_result:
	    return message
    else:
        return error_resp(2, "LACK_PARAM ERROR")
    file_path1 = os.path.join(app.config['FEATURE_FOLDER'], "{}.bin".format(id_0))
    file_path2 = os.path.join(app.config['FEATURE_FOLDER'], "{}.bin".format(id_1))
    if(not os.path.exists(file_path1) or not os.path.exists(file_path2)):
	return error_resp(2, "FACE_ID ERROR")

    feat1 = load_mat(file_path1)
    feat2 = load_mat(file_path2)
    score = eng.calc_score_result(feat1, feat2)
    resp = jsonify(status=1, result="Same Person" if score>0.5 else "Different Person", similarity="%.3f" % score)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def inner_recognize(feature):
    global cele_features
    global cele_names
    scores = []
    names = []
    for i in xrange(len(cele_features)):
        cele_feat = cele_features[i]
	cele_name = cele_names[i]
	score = eng.calc_score_result(feature, cele_feat)
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

@app.route('/api/recognize', methods=['POST'])
def recognize():
    # verify the keys
    ACCESS_KEY = request.form.get('ACCESS_KEY').encode("utf-8")
    TIMESTAMP = request.form.get('TIMESTAMP').encode("utf-8")
    SIGN_KEY = request.form.get('SIGN_KEY').encode("utf-8")
    user_id = request.form.get('user_id').encode("utf-8")
    id = request.form.get('id').encode("utf-8")
    if ACCESS_KEY and TIMESTAMP and SIGN_KEY and user_id >= 0 and id >= 0:
        check_result, message = _check_security(ACCESS_KEY,TIMESTAMP,SIGN_KEY,user_id)
	if not check_result:
	    return message
    else:
        return error_resp(2, "LACK_PARAM ERROR")
    
    file_path = os.path.join(app.config['FEATURE_FOLDER'], "{}.bin".format(id))
    if(not os.path.exists(file_path)):
	return error_resp(2, "FACE_ID ERROR")

    feat = load_mat(file_path)
    names, scores = inner_recognize(feat)
    min_value = min(scores)
    scores = [s-min_value for s in scores]
    sum_value = sum(scores)
    norm_scores = ["%.2f" % np.divide(s,sum_value) for s in scores]
    results = []
    for i in xrange(len(norm_scores)):
	results.append(serialize3(names[i], norm_scores[i]))
    resp = jsonify(status=1, results=results)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def inner_detect_politician(feature):
    global polit_features
    global polit_names
    max_score = 0.0
    match_name = ""
    for i in xrange(len(polit_features)):
        polit_feat = polit_features[i]
	polit_name = polit_names[i]
	score = eng.calc_score_result(feature, polit_feat)
	if(score > max_score):
	    max_score = score
	    match_name = polit_name

    if(max_score > 0.62):
	return match_name, max_score
    else:
	return None, 0

def serialize4(name, rect, score):
    return {"name": name,
	    "rect": rect.tolist(),
	    "score": "%.2f" % score, }

def politician_detect_in_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    assert(np.prod(image.shape) > 0)
    # Begin to process
    rects, shapes = eng.calc_landmark_result(image)

    if(len(rects) == 0):
	return []
    # Calc features and store them
    features = eng.calc_feature_result(image, shapes)

    detected_samples = []
    for i in xrange(len(features)):
	detected_politician, score = inner_detect_politician(features[i,:])
	if(detected_politician is not None):
	    detected_samples.append(serialize4(detected_politician, rects[i], score))
    return detected_samples

def politician_detect_in_video(filename):
    # split video frame
    frame_root = "/tmp/polit_detect_temp_frames_%s" % filename
    os.makedirs(frame_root)
    shell_script = 'ffmpeg -i {} -vf fps=1 {}/%04d.png'.format(filename, frame_root)
    os.system(shell_script)

    results = []
    for index, frame_name in enumerate(os.listdir(frame_root)):
        time_in_sec = int(frame_name.split(".png")[0])
        frame = cv2.imread(os.path.join(frame_root, frame_name), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rects, shapes = eng.calc_landmark_result(frame)
	if(len(rects) == 0):
	    continue
    
        features = eng.calc_feature_result(frame, shapes)

        # Here is the detected result from only one frame
        detected_samples = []
        for j in xrange(len(features)):
            detected_politician, score = inner_detect_politician(features[j,:])
            if(detected_politician is not None):
                detected_samples.append(serialize4(detected_politician,rects[j],score))
        if(len(detected_samples) != 0):
            results.append( {"samples": detected_samples, "time":time_in_sec} )

    # remove the temp frame directory
    shutil.rmtree(frame_root)
    return results

def politician_detect_thread(url, task_id, TIMESTAMP, is_video):
    print("start thread %s" % threading.currentThread().getName())
    f = urlopen(url)
    file_type = f.info().type
    filename = os.path.join('/tmp', "temp_%s.%s" % (task_id, file_type.split('/')[1]))
    with open(filename, "wb") as local_f:
	local_f.write(f.read())
    file_size = os.path.getsize(filename)

    assert(is_video == (file_type.split('/')[0] == 'video'))
    if not is_video:
        # image-type will call one-detection for image
        result = politician_detect_in_image(filename)
	os.remove(filename)
        resp = json.dumps({"task_status": 1,
			"task_id":task_id,
			"video_url":url,
			"video_size":file_size,
			"timestamp":TIMESTAMP,
			"video_duration":" ",
			"video_results":result}, ensure_ascii=False,indent=2)
        timestamp = int(time.time()) - time_error
        user_id = 1
        key_params = [("ACCESS_KEY", bm_acc_key),("TIMESTAMP", timestamp),("user_id", int(user_id))]    
        sign_key = _sign(bm_sec_key, key_params)
        payload = {"result": resp,
                   "ACCESS_KEY": bm_acc_key,
                   "SIGN_KEY": sign_key,
                   "user_id": user_id,
                   "TIMESTAMP": timestamp}
        print("%s return:" % threading.currentThread().getName())
        r = requests.post(recall_url, data=payload)    
        return 
    else:
        # video-type will call one-detection every second for frame
	metadata = skvideo.io.ffprobe(filename)
	frame_ = metadata['video']['@avg_frame_rate'].split('/')
	nb_frames = int(metadata['video']['@nb_frames'])
	framerate = int(math.floor(float(frame_[0]) / float(frame_[1])))
	video_len = nb_frames / framerate + 1
	result = politician_detect_in_video(filename)
	os.remove(filename)
        resp = json.dumps({"task_status": 1,
			"task_id":task_id,
			"video_url":url,
			"video_size":file_size,
			"timestamp":TIMESTAMP,
			"video_duration":video_len,
			"video_results":result}, ensure_ascii=False,indent=2)
        timestamp = int(time.time()) - time_error
        user_id = 1
        key_params = [("ACCESS_KEY", bm_acc_key),("TIMESTAMP", timestamp),("user_id", int(user_id))]
        sign_key = _sign(bm_sec_key, key_params)
        payload = {"result": resp,
                   "ACCESS_KEY": bm_acc_key,
                   "SIGN_KEY": sign_key,
                   "user_id": user_id,
                   "TIMESTAMP": timestamp}
        print("%s return:" % threading.currentThread().getName())
        r = requests.post(recall_url, data=payload)
	return

@app.route('/api/politician_detect', methods=['POST'])
def politician_detect():
    url = request.form.get('url').encode('utf-8')
    # actually this is not needed, I add this only because baoming's interface will call like this
    TIMESTAMP = request.form.get('TIMESTAMP').encode('utf-8')
    task_id = request.form.get('task_id').encode('utf-8')
   
    # check format type and download the file
    print url
    f = urlopen(url)
    #size = f.headers['content-length']
    file_type = f.info().type
    is_video = True
    if file_type.split('/')[0] == 'image':
        is_video = False
	if file_type.split('/')[1] not in ALLOWED_EXTENSIONS:
	    return error_resp(3, "FORMAT ERROR")
    else:
	if file_type.split('/')[1] not in ALLOWED_VIDEO_EXTENSIONS:
	    return error_resp(3, "FORMAT ERROR")
    
    task_name = "task_thread_%s" % task_id
    task_pool.append(threading.Thread(target=politician_detect_thread,name=task_name,args=(url,task_id,TIMESTAMP,is_video) )) 
    task_pool[-1].start()
    #task_pool[-1].join()
    resp = jsonify(task_status=1,message="Received the url")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

    	
	
        
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # norm index visit
    return render_template('index.html')

def load_cele_features(celefeature_folder):
    features = []
    names = []
    for the_file in os.listdir(celefeature_folder):
	name = the_file.rsplit('#', 1)[0]
	file_path = os.path.join(celefeature_folder, the_file)
	feature = load_mat(file_path)
        features.append(feature)
	names.append(name)
    return features, names

if __name__=='__main__':
    face_id = 0
    max_face_id_num = 1000
    time_error = 28384
    #time_error = 0
    # Params
    acc_key = "your access key"
    sec_key = "your secure key"
    bm_acc_key = "your return access key"
    bm_sec_key = "your return secure key"
    recall_url = "your return back api"
    model_path = "./backend_api/save_model"
    config_path = "./backend_api/save_model/config.json"

    eng = FACE_engine(model_path, config_path)
    # Clear feature files
    feature_folder = app.config['FEATURE_FOLDER']
    for the_file in os.listdir(feature_folder):
        file_path = os.path.join(feature_folder, the_file)
	try:
	    if os.path.isfile(file_path):
	        os.unlink(file_path)
	except Exception as e:
	    print(e)
    
    # Load all celebrity features
    cele_features, cele_names = load_cele_features(app.config['CELEPOOL_FOLDER'])
    polit_features, polit_names = load_cele_features(app.config['BLACKPOOL_FOLDER'])
    task_pool = list()
    app.run(host='0.0.0.0', port=63001)
