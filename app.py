# app.py
import streamlit as st
import tempfile, cv2, os, numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Deepfake Video Detector", layout="centered")

st.title("Deepfake Video Detector — Demo")
st.markdown("Upload a video and the app will analyze faces frame-by-frame.")

uploaded = st.file_uploader("Upload video (mp4, mov)", type=["mp4","mov","avi"])
model_load_state = st.text("Loading model...")
# replace 'model.h5' with your trained model path
try:
    model = load_model("model.h5")  # Xception-based or other Keras model
    model_load_state.text("Model loaded.")
except Exception as e:
    model = None
    model_load_state.text(f"Model load failed: {e}")

detector = MTCNN()

def extract_frames(video_path, sample_rate=3):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % sample_rate == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames

def detect_and_crop_faces(frames, size=(299,299)):
    crops = []
    for f in frames:
        res = detector.detect_faces(f)
        if res:
            # take largest face
            res_sorted = sorted(res, key=lambda x: x['box'][2]*x['box'][3], reverse=True)
            x,y,w,h = res_sorted[0]['box']
            x, y = max(0,x), max(0,y)
            crop = f[y:y+h, x:x+w]
            crop = cv2.resize(crop, size)
            crops.append(crop)
        else:
            crops.append(None)
    return crops

if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())
    video_path = tfile.name

    st.info("Extracting frames...")
    frames = extract_frames(video_path, sample_rate=3)  # adjust sample_rate
    st.write(f"Extracted {len(frames)} frames (sampled).")

    st.info("Detecting faces...")
    crops = detect_and_crop_faces(frames)
    st.write("Running model inference...")

    frame_scores = []
    for idx, crop in enumerate(crops):
        if crop is None:
            frame_scores.append(None)
            continue
        x = img_to_array(crop)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # for Xception
        if model is None:
            score = 0.5  # placeholder
        else:
            pred = model.predict(x, verbose=0)[0][0]
            score = float(pred)  # assume softmax/sigmoid -> probability fake
        frame_scores.append(score)

    # Aggregate to video-level score (mean of frame scores ignoring None)
    valid_scores = [s for s in frame_scores if s is not None]
    if valid_scores:
        video_score = float(np.mean(valid_scores))
        verdict = "Fake" if video_score >= 0.5 else "Real"
    else:
        video_score = None
        verdict = "No face detected"

    st.subheader("Result")
    if video_score is not None:
        st.metric("Fake probability (video-level)", f"{video_score:.3f}", help="Mean of per-face probabilities")
        st.write("Final verdict:", verdict)
    else:
        st.write(verdict)

    # show thumbnails and per-frame scores
    st.subheader("Frames (sampled) with scores")
    cols = st.columns(3)
    for i, (frame, score) in enumerate(zip(frames, frame_scores)):
        col = cols[i % 3]
        if score is None:
            col.image(frame, caption=f"Frame {i} — no face", use_column_width=True)
        else:
            col.image(frame, caption=f"Frame {i} — fake p={score:.3f}", use_column_width=True)

    # cleanup
    os.unlink(video_path)
