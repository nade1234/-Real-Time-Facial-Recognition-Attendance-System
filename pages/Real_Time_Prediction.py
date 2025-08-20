# Real_Time_Prediction.py

import os
import time
import streamlit as st
import av
import face_rec  # votre module existant
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoHTMLAttributes,
)
from aiortc.contrib.media import MediaRecorder

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='Real-Time Prediction')

st.subheader('Real-Time Attendance System')

# ─── Retrieve face DB from Redis ───────────────────────────────────────────────
with st.spinner('Retrieving data from Redis DB...'):
    redis_face_db = face_rec.retrieve_data(name='academy:register')
    if redis_face_db is not None and not redis_face_db.empty:
        display_data = redis_face_db[['Name', 'Role']]
        st.dataframe(display_data)
    else:
        st.warning("Aucune donnée trouvée dans la base Redis")

# ─── Prepare recording ──────────────────────────────────────────────────────────
VIDEO_FILE = os.path.join(os.getcwd(), "attendance_recording.mp4")
recorder_holder = []

def recorder_factory() -> MediaRecorder:
    """Create & hold onto the MediaRecorder so we can stop it later."""
    rec = MediaRecorder(VIDEO_FILE, format="mp4")
    recorder_holder.append(rec)
    return rec

# ─── Real-time predictor instance & timing ──────────────────────────────────────
waitTime = 30
setTime = time.time()
realtimepred = face_rec.RealTimePred()

# ─── Video frame callback ───────────────────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global setTime
    img = frame.to_ndarray(format="bgr24")

    # Predict & annotate
    pred_img = realtimepred.face_prediction(
        img,
        redis_face_db,
        feature_column='facial_features',
        name_role=['Name', 'Role'],
        thresh=0.5
    )

    # Save logs periodically
    if time.time() - setTime >= waitTime:
        realtimepred.savelogs_redis()
        setTime = time.time()

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# ─── Launch WebRTC with bigger preview & recording ─────────────────────────────
webrtc_ctx = webrtc_streamer(
    key="real-time-attendance",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    in_recorder_factory=recorder_factory,
    media_stream_constraints={"video": True},
    video_html_attrs=VideoHTMLAttributes(
        autoPlay=True,
        controls=False,
        muted=True,
        style={
            "width": "640px",    # ← larger preview
            "height": "480px",
            "border": "2px solid #444",
            "border-radius": "4px",
        },
    ),
)

# ─── When the user stops the stream, finalize & replay ─────────────────────────
if webrtc_ctx.state.playing is False and recorder_holder:
    # Stop all recorders to flush the file
    for rec in recorder_holder:
        try:
            rec.stop()
        except Exception:
            pass
    recorder_holder.clear()

    # Then replay if the file exists
    if os.path.isfile(VIDEO_FILE):
        st.markdown("### Replay of attendance recording")
        st.video(VIDEO_FILE)
    else:
        st.error(
            "⚠️ Recording not found. Ensure `ffmpeg` is installed and on your PATH, "
            "and that this script has write permissions in its working directory."
        )
