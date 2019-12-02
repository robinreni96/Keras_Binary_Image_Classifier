from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping,ReduceLROnPlateau, CSVLogger
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()

config.gpu_options.allow_growth = True

tf.keras.backend.set_session(tf.Session(config=config))

print(K.tensorflow_backend._get_available_gpus())
print("------------------------------->>>>>>>>>>>>>>>>>>>")

batch_size_train = 32
batch_size_val = 16
num_classes= 2
classes_required = ["pookalam","random"]
# intereseted_folder='Documents'
STANDARD_SIZE=(224,224)

# hyper parameters for model
nb_classes = 2  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 224, 224  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation

datagen=ImageDataGenerator(data_format=K.image_data_format())

train_path = 'data/train/'
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=classes_required, batch_size=batch_size_train)

val_path = 'data/val/'
val_batches = ImageDataGenerator().flow_from_directory(val_path, target_size=(224,224), classes=classes_required, batch_size=batch_size_val)

base_model = ResNet50(input_shape=(img_height,img_width,3), weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output

# Top Model Block
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nb_classes, activation='softmax')(x)

# add your top layer block to your base model
model = Model(base_model.input, predictions)
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#callback = [,ModelCheckpoint( "model_best.h5",save_best_only=True)]
callbacks = [
		ModelCheckpoint(filepath="model/model_best.h5", monitor="val_acc", save_best_only=True),
		ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5),EarlyStopping(patience=5)
	]
model.fit_generator(train_batches,steps_per_epoch=2000//batch_size_train,
        epochs=20,
        validation_data= val_batches,
        validation_steps=1000// batch_size_val ,callbacks=callbacks,verbose=1)


model.save("model_last.h5")
print("Model trained Successfully")