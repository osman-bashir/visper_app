import streamlit as st
import tempfile
import pipeline
from  preprocess import save2vid ,preprocess_video
# Set the title of the app
st.title("Visual Speech Recognition")

# Add a header
st.header("Upload your MP4 video file")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an MP4 video", type=["mp4"])

if uploaded_file is not None:
    # Display the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_file = temp_file.name
else:
    video_file = "download.mp4"


#Predict The Text
dst_filename = preprocess_video(src_filename= video_file, dst_filename="roi.mp4")
st.video("roi.mp4")
pipeline = pipeline.build_pipeline()
result = pipeline("roi.mp4",3)
st.text("You said: " + result)


