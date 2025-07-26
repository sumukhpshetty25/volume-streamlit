import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
from HandTrackingMin import handDetector  # ✅ Use your custom minimal module

st.title("🖐️ Volume Hand Control (Streamlit Demo)")
st.markdown("Upload an image of a hand showing your thumb and index finger, and this app simulates volume control based on finger distance.")

uploaded_file = st.file_uploader("Upload a hand image (JPG or PNG)", type=["jpg", "png"])
detector = handDetector(detectionCon=0.7)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (640, 480))

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        volPer = int(np.interp(length, [25, 200], [0, 100]))

        st.metric("Finger Distance", f"{int(length)} px")
        st.metric("Simulated Volume", f"{volPer} %")
        st.progress(volPer)
    else:
        st.warning("No hand landmarks detected. Try another image.")

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Processed Image', use_column_width=True)
else:
    st.info("Upload an image with a visible hand to get started.")
