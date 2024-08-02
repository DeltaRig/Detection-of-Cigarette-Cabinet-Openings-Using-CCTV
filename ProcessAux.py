##########################################################################################
# Imports
##########################################################################################
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras


def main(path):
  lstNames  = []
  lstImages = []

  for filename in os.listdir(path):
    img = keras.preprocessing.image.load_img(os.path.join(path,filename), target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    
    lstImages.append(img)
    lstNames.append(str(os.path.join(path,filename)))
    print('added: ', filename)

  
  lstImages = keras.applications.vgg19.preprocess_input(np.array(lstImages))
  nIndex = 0
  while nIndex < len(lstNames):
    cv2.imwrite(lstNames[nIndex], lstImages[nIndex])
    print('saved: ', lstNames[nIndex])
    nIndex+=1

main('open') # replace for other path if your videos are in other folders
main('close')