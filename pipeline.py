import os
import torch
import torchaudio
import torchvision
from lightning_vsr import ModelModule
from datamodule.av_dataset import cut_or_pad
from datamodule.transforms import AudioTransform, VideoTransform
import numpy as np
import torch.multiprocessing as mp
from subprocess import CalledProcessError, run
from hydra import compose, initialize

class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="mediapipe"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        self.multi_lang = cfg.data.multi_lang
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")
        if self.modality in ["video", "audiovisual"]:
            self.video_transform = VideoTransform(subset="test")

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.eval()


    def forward(self, data_filename, lang):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality in ["audio", "audiovisual"]:
            try:
                audio, sample_rate = self.load_audio(data_filename)
            except:
                if os.path.isfile(data_filename.replace('.mp4', '.wav')):
                    audio_filename = data_filename.replace('.mp4', '.wav')
                    audio, sample_rate = self.load_audio(audio_filename)
                else:
                    raise ValueError(f"Unable to load audio for {data_filename}")
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)

        if self.modality in ["video", "audiovisual"]:
            video = self.load_video(data_filename)
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video)

        if self.modality == "video":
            with torch.no_grad():
                transcript = self.modelmodule(video, lang=lang)
        elif self.modality == "audio":
            with torch.no_grad():
                transcript = self.modelmodule(audio)
        elif self.modality == "audiovisual":
            rate_ratio = len(audio) // len(video)
            if rate_ratio > 670 or rate_ratio < 530:
                print(f"WARNING: Inconsistent frame ratio for {data_filename}. Found audio length: {len(audio)}, video length: {len(video)}. It might affect the performance.")
            if rate_ratio != 640:
                audio = cut_or_pad(audio, len(video) * 640)
            with torch.no_grad():
                transcript = self.modelmodule(video, audio, lang=lang)

        return transcript


    def load_audio(self, data_filename: str, sr: int = 16000):
        """
        Sourced from https://github.com/openai/whisper/blob/main/whisper/audio.py
        Open an audio file and read as mono waveform, resampling as necessary

        Parameters
        ----------
        file: str
            The audio file to open

        sr: int
            The sample rate to resample the audio if necessary

        Returns
        -------
        A NumPy array containing the audio waveform, in float32 dtype.
        """

        # This launches a subprocess to decode audio while down-mixing
        # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
        # fmt: off
        cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", data_filename, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
        # fmt: on
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        waveform = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        waveform = torch.FloatTensor(waveform).unsqueeze(0)

        return waveform, sr

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
def build_pipeline ():
    with initialize(version_base=None, config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config")
      # Remove GPU-specific settings
    pipeline = InferencePipeline(cfg)
    
    # Ensure model is moved to CPU
    pipeline.modelmodule = pipeline.modelmodule.to('cpu')
    return pipeline 
