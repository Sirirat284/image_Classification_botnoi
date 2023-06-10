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

import time

styles = {'material-icons':{'color': 'green'},
          'text-icon-link-close-container': {'box-shadow': '#3896de 0px 10px'},
          'notification-text': {'':''},
          'close-button':{'':''},
          'link':{'':''}}


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
def prediction_from_img(image_data,clf):
  featList = []
  cvo = encode_img_to_vec(image_data)
  featList.append(cvo)
  dat = pd.DataFrame(data=[featList]).T
  return clf.predict(np.vstack(dat[0].values))[0]

@st.cache_data
def load_data():
    return pd.DataFrame(
        {
            "หมา🐶": ["ชิสุ","โกลเด้นท์ รีทรีฟเวอร์","บีเกิ้ล","ชิวาว่า" , "เฟรนช์บูลด๊อก" , "ไซบีเรียน ฮัสกี้"],
            "แมว🐱": ["สก๊อตติช โฟลด์", "เอ็กโซติก", "อเมริกัน ชอร์ตแฮร์", "วิเชียรมาศ" ,"ขาวมณี","สีสวาด"]
        }
    )
class main :
  def main():
    print("server start")

    st.title("Image classification")
    st.subheader("🐱คัดแยกสายพันธุ์ หมา&แมว ยอดฮิตในไทย🐶")
    mp4_file = "https://ik.imagekit.io/seproject/converted_video.mp4?updatedAt=1686420832938"

    if mp4_file is not None:
      # แสดงวิดีโอ
      st.video(mp4_file,format='GIF', start_time=0)

    st.write("📣สายพันธุ์ หมา&แมว ที่ AI จำแนกได้")
    df = load_data()
    st.dataframe(df )
    st.subheader("⚠️ในการจำแนกสายพันธุ์ สามารถ copy image address url หรือ จะเป็นการ upload file ก็ได้")
    url = st.text_input('📌URL รูปภาพที่ท่านต้องการจะทดสอบ (ต้องใช้ Image address หรือ ที่อยู่ของรูปภาพ และ ต้องลงท้ายด้วย.jpg หรือ .png)','Image address url( .jpg or .png )')
    model = pickle.load(open('./model/vectorizer.pickle', 'rb'))
    if st.button('predict'):
      with st.spinner('กำลังประมวลผลรูปภาพ'):
         time.sleep(2)
      pr = prediction_from_URL(url,model)
      print(pr)
      st.image(url, caption='รูปภาพที่อัปโหลด')
      st.subheader("✅ รูปนี้เป็นสายพันธ์ุ : "+str(pr))
      st.write("ต้องขออภัยถ้าหาก AI ให้คำตอบที่ผิดพลาด นี่เป็นเพียงแค่ส่วนนึงในการเรียนการสอน")
    uploaded_file = st.file_uploader('อัปโหลดรูปภาพ', type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        # ดึงข้อมูลจากไฟล์รูปภาพที่อัปโหลด
        image = Image.open(uploaded_file)
        # แสดงรูปภาพใน Streamlit
        with st.spinner('กำลังประมวลผลรูปภาพ'):
         time.sleep(1.5)
        st.image(image, caption='รูปภาพที่อัปโหลด')
        # แปลงรูปภาพเป็น bytearray
        byte_stream = io.BytesIO()
        image.save(byte_stream, format='JPEG')
        image_data = byte_stream.getvalue()
        pr = prediction_from_img(image_data,model)
        print(pr)
        st.subheader("✅ รูปนี้เป็นสายพันธ์ุ : "+str(pr))
        st.write("ต้องขออภัยถ้าหาก AI ให้คำตอบที่ผิดพลาด นี่เป็นเพียงแค่ส่วนนึงในการเรียนการสอน")
      
    # st.subheader("ใส่ข้อมูลของคุณที่ต้องการจะเทรน AI")
    # size_of_prediction = st.slider('จำนวนชุดข้อมูลต้องการเทรน AI (ยิ่งจำนวนชุดข้อมูลในการเทรน AI มีจำนวนมาก อาจจะส่งผลเวลาในการเทรน AI ก็จะนานขึ้น) ', 2, 5, 2)
    # data_list=[] 
    # if size_of_prediction == 2:
    #       data1 = st.text_input('ข้อมูลที่ต้องการชุดที่ 1', 'สุนัข')
    #       data2 = st.text_input('ข้อมูลที่ต้องการชุดที่ 2', 'แมว')
    #       data_list.append(data1)
    #       data_list.append(data2)
    # elif size_of_prediction == 3:
    #       data1 = st.text_input('ข้อมูลที่ต้องการชุดที่ 1', '')
    #       data2 = st.text_input('ข้อมูลที่ต้องการชุดที่ 2', '')
    #       data3 = st.text_input('ข้อมูลที่ต้องการชุดที่ 3', '')
    #       data_list.append(data1)
    #       data_list.append(data2)
    #       data_list.append(data3)
    # elif size_of_prediction == 4:
    #       data1 = st.text_input('ข้อมูลที่ต้องการชุดที่ 1', '')
    #       data2 = st.text_input('ข้อมูลที่ต้องการชุดที่ 2', '')
    #       data3 = st.text_input('ข้อมูลที่ต้องการชุดที่ 3', '')
    #       data4 = st.text_input('ข้อมูลที่ต้องการชุดที่ 4', '')
    #       data_list.append(data1)
    #       data_list.append(data2)
    #       data_list.append(data3)
    #       data_list.append(data4)
    # elif size_of_prediction == 5:
    #       data1 = st.text_input('ข้อมูลที่ต้องการชุดที่ 1', '')
    #       data2 = st.text_input('ข้อมูลที่ต้องการชุดที่ 2', '')
    #       data3 = st.text_input('ข้อมูลที่ต้องการชุดที่ 3', '')
    #       data4 = st.text_input('ข้อมูลที่ต้องการชุดที่ 4', '')
    #       data5 = st.text_input('ข้อมูลที่ต้องการชุดที่ 5', '')
    #       data_list.append(data1)
    #       data_list.append(data2)
    #       data_list.append(data3)
    #       data_list.append(data4)
    #       data_list.append(data5)
    
    # feat = None 
    # if st.button('train model'):
    #     with st.spinner('กำลังเทรน AI อาจจะใช้เวลานาน โปรดรอสักครู่ '):
    #       feat =featextraction(data_list )
    #       model =trainmodel(feat)
    #       custom_notification_box(icon='done', textDisplay='Train model สำเร็จ', externalLink='', url='#', styles=styles, key="foo")
    # if st.button('test model'):
    #     st.session_state["page"] = "หน้าสอง"
    # if st.button("กลับไปยังหน้าหลัก"):
    #       st.session_state["page"] = "หน้าหลัก"
  # Specify the number of columns

  def predict_page():
    if st.button('back'):
      st.session_state["page"] = "หน้าหลัก"
    st.title("Image classification")
    # st.subheader("ทดสอบ AI ")
    # st.write("ในการทดสอบ สามารถ copy image address url หรือ จะเป็นการ upload file ก็ได้")
    # url = st.text_input('URL รูปภาพที่ท่านต้องการจะทดสอบ (ต้องใช้ Image address หรือ ที่อยู่ของรูปภาพ และ ต้องลงท้ายด้วย.jpg หรือ .png)','Image address url( .jpg or .png )')
    # model = pickle.load(open('./model/vectorizer.pickle', 'rb'))
    # if st.button('predict'):
    #   pr = prediction_from_URL(url,model)
    #   print(pr)
    #   st.image(url, caption='รูปภาพที่อัปโหลด')
    #   st.subheader("รูปนี้เป็นรูป : "+str(pr))
    # uploaded_file = st.file_uploader('อัปโหลดรูปภาพ', type=['jpg', 'png', 'jpeg'])
    # if uploaded_file is not None:
    #     # ดึงข้อมูลจากไฟล์รูปภาพที่อัปโหลด
    #     image = Image.open(uploaded_file)
    #     # แสดงรูปภาพใน Streamlit
    #     st.image(image, caption='รูปภาพที่อัปโหลด')
    #     # แปลงรูปภาพเป็น bytearray
    #     byte_stream = io.BytesIO()
    #     image.save(byte_stream, format='JPEG')
    #     image_data = byte_stream.getvalue()
    #     pr = prediction_from_img(image_data,model)
    #     print(pr)
    #     st.subheader("รูปนี้เป็นรูป : "+str(pr))

  if "page" not in st.session_state:
      st.session_state["page"] = "หน้าหลัก"

  if st.session_state["page"] == "หน้าสอง":
      predict_page()
  else:
      main()


if __name__ == "__main__":
    main()

