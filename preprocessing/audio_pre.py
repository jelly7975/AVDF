import os
import numpy as np
from moviepy.editor import AudioFileClip
import librosa
import soundfile as sf


def audio (root, target_time=5):
    for action in os.listdir(root):
        action_path = os.path.join(root, action)
        for filename in os.listdir(action_path):
            file_path = os.path.join(action_path, filename)

            if filename.endswith('.mp4'):
                wav_filename = os.path.splitext(filename)[0] + ".wav"
                wav_path = os.path.join(action_path, wav_filename)

                try:
                    audio = AudioFileClip(file_path)
                    audio.write_audiofile(wav_path)
                    audio.close()
                    file_path = wav_path  # 后续处理WAV文件
                    filename = wav_filename
                except Exception as e:
                    print(f"Error converting {file_path}: {e}")
                    continue

            if filename.endswith('.wav') and '_cropped' not in filename:
                cropped_filename = filename[:-4] + '_cropped.wav'
                cropped_path = os.path.join(action_path, cropped_filename)
                try:
                    y, sr = librosa.load(file_path, sr=22050)
                    target_length = int(sr * target_time)
                    if len(y) < target_length:
                        y = np.pad(y, (0, target_length - len(y)))
                    else:
                        remain = len(y) - target_length
                        start = remain // 2
                        y = y[start:start + target_length]
                    sf.write(cropped_path, y, sr)
                    print(f"Processed: {action}/{cropped_filename}")
                except Exception as e:
                    print(f"Error cropping {file_path}: {e}")

# 使用示例
root = ''  # 路径
aduio (root, target_time=5)  # 默认5秒