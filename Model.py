from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import numpy as np
from keras.callbacks import ModelCheckpoint

data=np.load('data.npy')
target=np.load('target.npy')

print(data.shape)

baseModel = Xception(weights="imagenet", include_top=False,
	input_tensor=Input(data.shape[1:]))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=0.0001, decay=1e-4 / 20)
model.compile(loss="binary_crossentropy", optimizer='Adam',
	metrics=["accuracy"])

model.summary()
from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)


model_saved=model.fit(train_data,train_target,epochs=7,validation_split=0.2)
model.save('mymodel.h5',model_saved)