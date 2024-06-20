import os
import cv2
import torchvision
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector

def save2vid(filename, vid, frames_per_second):
    torchvision.io.write_video(filename, vid, frames_per_second)

def preprocess_video(src_filename, dst_filename):
    dataloader = AVSRDataLoader(modality="video", speed_rate=1, transform=False, detector="mediapipe", convert_gray=False)
    landmarks_detector = LandmarksDetector()
    landmarks = landmarks_detector(src_filename)
    data = dataloader.load_data(src_filename, landmarks)
    fps = cv2.VideoCapture(src_filename).get(cv2.CAP_PROP_FPS)
    save2vid(dst_filename, data, fps)
    return dst_filename

