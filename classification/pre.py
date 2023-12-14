#!/usr/bin/env python
# coding: utf-8

# In[16]:

import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# 저장된 모델 불러오기
loaded_model = load_model("/home/user/pr/prapp/kumoh1.h5")

dir_name = ['d1_1', 'd1_2', 'd1_3', 'db_1', 'db_2', 'db_3', 'etc', 'foun']

def predict(path):
    image_path = path  # 예측하고자 하는 이미지의 경로를 입력하세요.
    img = image.load_img(image_path, target_size=(400, 400))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 이미지를 [0, 1] 범위로 정규화
    # 모델에 입력 이미지 전달하여 예측 수행
    predictions = loaded_model.predict(img_array)

    # 예측 결과 출력
    predicted_class = np.argmax(predictions)

    predicted_label = dir_name[predicted_class]

    if predicted_label == "etc":
        return "B123"
    return predicted_label

