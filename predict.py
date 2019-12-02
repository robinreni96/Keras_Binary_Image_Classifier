import os 
import numpy as np 
from keras.models import load_model
from keras.preprocessing import image
import cv2
import time
import shutil
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()

config.gpu_options.allow_growth = True

tf.keras.backend.set_session(tf.Session(config=config))


# Loading Model
model_dir = "model/model_best.h5"
model = load_model(model_dir)
print(" >>>>>>>>>>>>> Model Loaded Successfully <<<<<<<<<<<<<<<<<<<< ")

test_dir = "test_dir/test_set/"
image_paths = [os.path.join(test_dir,fn) for fn in next(os.walk(test_dir))[2]]
# images = load_image(image_paths)

# Predict Image
def predict(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    classes = model.predict(img)
    return np.argmax(classes)

result = []
for i in image_paths :
    try:
        print(i)
        result.append(predict(i,model))
    except:
        continue

# Write the result to dir
def check(result_dir, class_name):
    for i in class_name:
        path = os.path.join(result_dir,i)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    
result_dir = "test_dir/results/"
class_name = ["Pookalam","Random"]
check(result_dir, class_name)
counter = 1
for (i,j) in zip(image_paths,result):
    ts = str(time.time())
    img = cv2.imread(i)
    save_path = class_name[j]+"_" + str(counter)+"_"+ ts +".jpg"
    save_path = os.path.join(result_dir,class_name[j],save_path)
    cv2.imwrite( save_path, img )
    # cv2.imshow(class_name[j],img)
    cv2.waitKey(0)
    counter += 1
cv2.destroyAllWindows()