import os
import streamlit as st
import numpy as np
import av
import face_rec  # votre module existant
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoHTMLAttributes
from aiortc.contrib.media import MediaRecorder

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Registration Form")
st.subheader("Registration Form")

# ─── Initialize your face-rec registration form ────────────────────────────────
registration_form = face_rec.RegistrationForm()

# ─── User inputs ────────────────────────────────────────────────────────────────
person_name = st.text_input("Name", placeholder="First & Last Name")
role = st.selectbox("Select your Role", ["Student", "Teacher"])

# ─── Where to save the video ────────────────────────────────────────────────────
VIDEO_FILE = os.path.join(os.getcwd(), "registration.mp4")

# ─── Recorder holder so we can stop it later ───────────────────────────────────
recorder_holder = []

# ─── Factory that creates & stores our MediaRecorder ───────────────────────────
def recorder_factory() -> MediaRecorder:
    rec = MediaRecorder(VIDEO_FILE, format="mp4")
    recorder_holder.append(rec)
    return rec

# ─── Frame callback: get embedding and draw ░───────────────────────────────────
def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    reg_img, embedding = registration_form.get_embedding(img)
    if embedding is not None:
        # Append to your embeddings file
        with open("face_embedding.txt", "ab") as f:
            np.savetxt(f, embedding)
    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")

# ─── Launch the webcam stream ───────────────────────────────────────────────────
webrtc_ctx = webrtc_streamer(
    key="registration",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_callback,
    in_recorder_factory=recorder_factory,
    media_stream_constraints={"video": True},
    video_html_attrs=VideoHTMLAttributes(
        autoPlay=True,
        controls=False,
        muted=True,
        style={
            "width": "640px",    # ← larger preview width
            "height": "480px",   # ← larger preview height
            "border": "2px solid #444",
            "border-radius": "4px",
        },
    ),
)

# ─── Submit button: save metadata in Redis ──────────────────────────────────────
if st.button("Submit"):
    res = registration_form.save_data_in_redis_db(person_name, role)
    if res is True:
        st.success(f"{person_name} registered successfully!")
    elif res == "name_false":
        st.error("Please enter a valid name.")
    elif res == "file_false":
        st.error("face_embedding.txt not found. Please refresh and retry.")

# ─── When the stream ends, finalize recording and replay ───────────────────────
if webrtc_ctx.state.playing is False and recorder_holder:
    # Stop any recorders we created
    for rec in recorder_holder:
        try:
            rec.stop()
        except Exception:
            pass
    recorder_holder.clear()

    # If ffmpeg succeeded, the file will exist
    if os.path.isfile(VIDEO_FILE):
        st.markdown("### Replay of your recording")
        st.video(VIDEO_FILE)
    else:
        st.error(
            "⚠️ Recording not found. Ensure `ffmpeg` is installed and on your PATH, "
            "and that this script has write permissions in its working directory."
        )
