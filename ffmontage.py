import os
import shutil
import glob
import sys
from tqdm import tqdm
from zipfile import ZipFile
import gdown
import subprocess
from datetime import datetime, timedelta
from moviepy.editor import VideoFileClip, concatenate_videoclips

from cv2 import imread, resize, VideoCapture, CAP_PROP_FRAME_COUNT, CAP_PROP_FPS
from np import array

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
            img = imread(image_path) / 255.
            img = resize(img, (299, 299))
        else:
            img = resize(image_path / 255., (299, 299))
        return img

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
    
    def time_to_str(self, timestr):
      ftr = [3600,60,1]
      return sum([a*b for a,b in zip(ftr, map(float,str(timestr).split(':')))])

    def video_process(self):
      self.download_video()
      if not os.path.exists(self.concat_dir):
          os.mkdir(self.concat_dir)
      cap = VideoCapture(self.video_path)
      total_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
      fps = cap.get(CAP_PROP_FPS)
      buffer_size = self.time_interval * fps
      current_frame_no = 0
      divisor = fps // 3
      seconds_per_frame = 1 / fps
      vid_no = 0
      bar = tqdm(total=total_frames)
      predict = self.model.predict
      input_image = self.input_image
      time_to_str = self.time_to_str
      current_time = datetime(1, 2, 3, hour=0, minute=0, second=0)
      time_interval = self.time_interval
      read = cap.read
      
      while cap.isOpened():
        try:
          ret, frame = read()
          if not ret:
            break
          current_time += timedelta(seconds=seconds_per_frame)
          if current_frame_no%divisor==0:
            image_in = array([input_image(frame)])
            flag = predict(image_in)[0][0] > 0.5
            if flag:
              i = 0
              while i < buffer_size :
                ret, frame = read()
                if not ret:
                  break
                i += 1
                current_frame_no += 1
                bar.update(1)
              start_time = max(time_to_str(current_time.time()) - time_interval, 0)
              end_time = time_to_str(current_time.time()) + time_interval
              movie = VideoFileClip('temp/download.mp4').subclip(start_time, end_time)
              movie.write_videofile(f"temp/to_concat/{vid_no}.mp4", verbose=False, progress_bar=False)
              bar.set_postfix_str(f'Partitions : {vid_no}')
              vid_no += 1
              current_time += timedelta(seconds=time_interval)
            else:
              current_frame_no += 1
              bar.update(1)
          else:
            current_frame_no += 1
            bar.update(1)
        except:
          bar.close()
          cap.release()
          shutil.rmtree('temp/')
          print(sys.exc_info())
          return
      bar.close()
      cap.release()
      now = datetime.today().strftime("Montage_%H_%M_%S")
      concat_file_name = f'{now}.mp4'
      with open('txt1.txt', 'a') as text_file:
          for file in sorted(glob.glob('temp/to_concat'+'/**'), key=lambda x: int(x.split('/')[-1].split('.')[0])):
            text_file.write('file '+file+'\n')
      print('Starting Concatanation...')
      subprocess.run(f'ffmpeg -f concat -safe 0 -i txt1.txt -c copy -preset slower {concat_file_name}', shell=True)
      print('Process Complete')
      os.remove('txt1.txt')
      shutil.rmtree('temp/')
        
