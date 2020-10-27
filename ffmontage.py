import os
import shutil
import glob
import sys
from tqdm import tqdm
from zipfile import ZipFile
import gdown
import subprocess
import datetime
import multiprocessing as mp

import cv2
import numpy as np

import tensorflow as tf


class FFMontage:
    def __init__(self, time_interval=2):
        self.video_path = 'temp/download.mp4'
        self.model_path = 'model'
        self.model = self.download_model()
        self.time_interval = time_interval
        self.concat_dir = 'temp/to_concat'
        self.partition_cmds = []
        self.txt_concat = 'to_concat'

    def download_model(self):
        url = 'https://drive.google.com/uc?id=18qrmcnwNXubyDizyddri_FSmIoyfuHK8'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
            output = f'{self.model_path}/model.zip'
            gdown.download(url, output, quiet=True)
            with ZipFile(f'{self.model_path}/model.zip') as zipf:
                os.mkdir(f'{self.model_path}/dir')
                zipf.extractall(f'{self.model_path}/dir')
        model = tf.keras.models.load_model(f'{self.model_path}/dir')
        return model

    def input_image(self, image_path):
        if type(image_path) == str:
            img = cv2.imread(image_path) / 255.
            img = cv2.resize(img, (299, 299))
        else:
            img = cv2.resize(image_path / 255., (299, 299))
        return img

    def is_true(self, image):
        image_in = np.array([self.input_image(image)])
        return self.model.predict(image_in)[0][0] > 0.5

    def download_video(self):
        if not os.path.exists('temp'):
            os.mkdir('temp')
        print("Make Sure that You have turned on Share with Everybody")
        link = input("Enter The G-Drive Link   ")
        if 'drive' in link:
            video_link = link.replace('file/d/', 'uc?id=').rstrip('/view?usp=sharing')
            gdown.download(video_link, self.video_path, quiet=True)
        else:
            print("Check Your Link")
        return

    def video_process(self):
        self.download_video()
        text_file = open('temp/text_file.txt', 'a')
        if not os.path.exists(self.concat_dir):
            os.mkdir(self.concat_dir)
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        partition_cmds = []
        buffer_size = self.time_interval * fps

        current_frame_no = 0
        divisor = fps // 3
        seconds_per_frame = 1 / fps
        vid_no = 0
        bar = tqdm(total=total_frames)

        current_time = datetime.datetime(1, 2, 3, hour=0, minute=0, second=0)
        end_time = datetime.datetime(1, 2, 3, hour=0, minute=0, second=0)
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                current_time += datetime.timedelta(seconds=seconds_per_frame)
                if current_frame_no % divisor == 0:
                    if self.is_true(frame):
                        end_time = current_time + datetime.timedelta(seconds=self.time_interval)
                        i = 0
                        while cap.isOpened() and i < self.time_interval * fps:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            current_frame_no += 1
                            bar.update(1)
                            i += 1
                        start_time = current_time - datetime.timedelta(seconds=2)
                        process_str = f'ffmpeg -i {self.video_path} -ss {start_time.time()} -to {end_time.time()} -c copy -preset ultrafast {self.concat_dir}/{str(vid_no)}.mp4'
                        subprocess.run([process_str], shell=True)
                        text_file.write(f'file {self.txt_concat}/{str(vid_no)}.mp4\n')
                        bar.set_postfix_str(f'Partitions : {vid_no}')
                        vid_no += 1
                        current_time = end_time
                    else:
                        current_frame_no += 1
                        bar.update(1)
                else:
                    current_frame_no += 1
                    bar.update(1)

            except:
                bar.close()
                cap.release()
                text_file.close()
                shutil.rmtree('temp/')
                print(sys.exc_info())
                return
        bar.close()
        cap.release()
        text_file.close()
        now = datetime.datetime.today().strftime("Montage_%D %H_%M_%S")
        concat_file_name = f'{now}.mp4'
        ffmpeg_cmd = f"ffmpeg -safe 0 -f concat -segment_time_metadata 1 -i temp/text_file.txt -vf select=concatdec_select -af aselect=concatdec_select,aresample=async=1 -preset ultrafast -max_muxing_queue_size 9999 " + concat_file_name
        subprocess.run([ffmpeg_cmd], shell=True)
        shutil.rmtree('temp/')
        
