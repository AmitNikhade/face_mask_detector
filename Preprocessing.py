
import cv2,os
from keras.utils import np_utils
import numpy as np

dataset=r'D:\BackupNew2020\Downloads\dataset'
files=os.listdir(dataset)
labels=[i for i in range(len(files))]
label_dict=dict(zip(files,labels))

size=224
data=[]
target=[]


for f in files:
    folder_path=os.path.join(dataset,f)
    images=os.listdir(folder_path)
        
    for image in images:
        image_path=os.path.join(folder_path,image)
        img=cv2.imread(image_path)
        try:
            col=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)           
            resized=cv2.resize(col,(size,size))
            data.append(resized)
            target.append(label_dict[f])
        except Exception as e:
            print(e)
    

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],size,size,3))
target=np.array(target)



new_target=np_utils.to_categorical(target)

np.save('data',data)
np.save('target',new_target)
