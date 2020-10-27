import os
import shutil

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


def download_model():
    url = 'https://drive.google.com/uc?id=18qrmcnwNXubyDizyddri_FSmIoyfuHK8'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    output = f'{model_path}/model.zip'
    gdown.download(url, output, quiet=True)
    with ZipFile(f'{model_path}/model.zip') as zipf:
        os.mkdir(f'{model_path}/dir')
        zipf.extractall(f'{model_path}/dir')
    model = tf.keras.models.load_model(f'{model_path}/dir')
    return model


def input_image(image_path):
    if type(image_path) == str:
        img = cv2.imread(image_path) / 255.
        img = cv2.resize(img, (299, 299))
    else:
        img = cv2.resize(image_path / 255., (299, 299))
    return img


def is_true(image):
    image_in = np.array([input_image(image)])
    return model.predict(image_in)[0][0] > 0.5


def download_video():
    if not os.path.exists('temp'):
        os.mkdir('temp')
    print("Make Sure that You have turned on Share with Everybody")
    link = input("Enter The G-Drive Link   ")
    if 'drive' in link:
        video_link = link.replace('file/d/', 'uc?id=').rstrip('/view?usp=sharing')
        gdown.download(video_link, video_path, quiet=True)
    else:
        print("Check Your Link")
    return


def video_part_mult(part_no):
    pbar = tqdm(total=len(partition_cmds))
    start = (len(partition_cmds) // num_processes) * part_no
    end = (len(partition_cmds) // num_processes) * (part_no + 1)
    for process_str in partition_cmds[start:end]:
        subprocess.run([process_str], shell=True)
        pbar.update(1)
    pbar.close()


def video_process():
    video_path = 'temp/download.mp4'
    model_path = 'model'
    model = download_model()
    time_interval = 2
    concat_dir = 'temp/to_concat'
    num_processes = mp.cpu_count()
    partition_cmds = []
    download_video()
    text_file = open('temp/text_file.txt', 'a')
    if not os.path.exists(concat_dir):
        os.mkdir(concat_dir)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

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
                if is_true(frame):
                    end_time = current_time + datetime.timedelta(seconds=time_interval)
                    i = 0
                    while cap.isOpened() and i < time_interval * fps:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        current_frame_no += 1
                        bar.update(1)
                        i += 1
                    start_time = current_time - datetime.timedelta(seconds=2)
                    process_str = f'ffmpeg -i {video_path} -ss {start_time.time()} -c:v libx264 -crf 18 -to {end_time.time()} -c:a copy -preset ultrafast {concat_dir}/{str(vid_no)}.mp4'
                    partition_cmds.append(process_str)
                    text_file.write(f'file {concat_dir}/{str(vid_no)}.mp4\n')
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
    p = mp.Pool(num_processes)
    p.map(video_part_mult, range(num_processes))

    now = datetime.datetime.today().strftime("Montage_%D %H_%M_%S")
    concat_file_name = f'{now}.mp4'
    concat_cmd = f"ffmpeg -y -loglevel error -f concat -safe 0 -i temp/text_file.txt -vcodec copy {concat_file_name}"
    subprocess.run([concat_cmd], shell=True)
    text_file.close()
    shutil.rmtree('temp/')
