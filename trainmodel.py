import streamlit as st
import pandas as pd
import numpy as np
import pickle
import get_image_url as get_img


import cv2
import requests
import io




from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.mobilenet import MobileNet

from tqdm import tqdm

from sklearn.svm import LinearSVC
import pickle

from streamlit_custom_notification_box import custom_notification_box

from PIL import Image


def encode_img_to_vec(image_data):
  # อ่านข้อมูลรูปภาพและแปลงเป็นรูปแบบที่ OpenCV ใช้งานได้
  image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
  image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
  resized_image = cv2.resize(image, (300, 300))
  # โหลดโมเดล VGG16 
  model = VGG16(weights='imagenet', include_top=False)

  # ปรับเป็นรูปแบบที่ใช้ในโมเดล VGG16
  preprocessed_image = preprocess_input(resized_image)
  expanded_image = np.expand_dims(preprocessed_image, axis=0)

  # ทำการ encode รูปภาพเป็นเวกเตอร์
  feature_vector = model.predict(expanded_image)

  # แปลงข้อมูลเป็นเวกเตอร์ 1 มิติ
  encoded_vector = feature_vector.flatten() 
  return encoded_vector

def encode(url):
  # ดาวน์โหลดรูปภาพจาก URL
  response = requests.get(url)
  image_data = response.content
  return encode_img_to_vec(image_data)

def featextraction(queryList ):
  dataList = []
  for j in range(len(queryList)):
    print(queryList[j])
    query = queryList[j]
    imgList = get_img.get_image_urls(query)
    featList = []
    for i in range(len(imgList)):
      try:
        img = imgList[i]
        cvo = encode(img)
        print("image ",i," success")
        featList.append(cvo)
      except:
        print("image ",i," Not success")
        pass
    dat = pd.DataFrame(data=[featList]).T
    dat['label'] = query
    dat.columns = ['feature','label']
    dataList.append(dat)  
  return pd.concat(dataList)

def trainmodel(res):
  print('start train')
  clf = LinearSVC()
  clf.fit(np.vstack(res['feature'].values),res['label'].values)
  path = "./model/vectorizer.pickle"
  with open(path, 'wb') as file:
    pickle.dump(clf, file)
    print('success')
  # return clf

def prediction_from_URL(imgurl,clf):
  featList = []
  cvo = encode(imgurl)
  featList.append(cvo)
  dat = pd.DataFrame(data=[featList]).T
  return clf.predict(np.vstack(dat[0].values))[0]

queryList = ['ชิสุ', 'โกลเด้นท์ รีทรีฟเวอร์' , 'บีเกิ้ล' , 'ชิวาว่า' , 'เฟรนช์บูลด๊อก' , 'ไซบีเรียน ฮัสกี้' , 'สก๊อตติช โฟลด์' , 'เอ็กโซติก' , 'อเมริกัน ชอร์ตแฮร์' , 'วิเชียรมาศ' , 'ขาวมณี','สีสวาด']
feat = featextraction(queryList)
# X = feat['feature']
# y = feat['label']
trainmodel(feat)
