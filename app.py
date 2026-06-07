
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import streamlit as st

model = keras.models.load_model("my_model (1).h5", compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("Drowsiness Detection")
nav_choice = st.sidebar.radio("Navigation", ("Home", "Sleep Detection", "Help Us Improve"), index=0)

if nav_choice == "Home":
    st.header("Prevents sleep deprivation road accidents, by alerting drowsy drivers.")
    st.image("ISHN0619_C3_pic.jpg")
    st.image("sleep.jfif", width=300)

elif nav_choice == "Sleep Detection":
    st.header("Drowsiness Detection")
    st.success("Take a photo below to check if you are drowsy.")
    photo = st.camera_input("Take a photo")
    if photo is not None:
        frame = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            st.warning("No face detected! Move closer to the camera.")
        else:
            decision = 0
            for (x, y, w, h) in faces:
                roi_color = frame[y:y+h, x:x+w]
                centx, centy = roi_color.shape[:2]
                centx //= 2
                centy //= 2
                try:
                    eye_1 = cv2.resize(roi_color[centy-40:centy, centx-70:centx], (86, 86))
                    eye_2 = cv2.resize(roi_color[centy-40:centy, centx:centx+70], (86, 86))
                    e1 = np.argmax(model.predict(np.expand_dims(eye_1, axis=0)))
                    e2 = np.argmax(model.predict(np.expand_dims(eye_2, axis=0)))
                    col1, col2, col3 = st.columns(3)
                    col1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Your photo")
                    col2.image(cv2.cvtColor(eye_1, cv2.COLOR_BGR2RGB), caption="Left eye")
                    col3.image(cv2.cvtColor(eye_2, cv2.COLOR_BGR2RGB), caption="Right eye")
                    if e1 != 1 and e2 != 1:
                        decision += 1
                except Exception as e:
                    st.warning(f"Could not extract eyes: {e}")
            if decision == 0:
                st.error("Eyes CLOSED - You appear DROWSY!")
            else:
                st.success("Eyes OPEN - You are ALERT!")
else:
    st.header("Help Us Improve")
    img_upload = st.file_uploader("Upload Image Here", ["png", "jpg", "jpeg"])
    if img_upload is not None:
        st.success("Uploaded Successfully! Thank you.")
