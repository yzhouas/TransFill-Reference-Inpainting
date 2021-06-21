from __future__ import unicode_literals
from pytube import YouTube
import os
import glob
import cv2
import youtube_dl
import numpy as np

def download_dataset(txt_dir, out_dir, sample_num=100, stride=1, remove_video=True):
    all_files = sorted(glob.glob(os.path.join(txt_dir, '*.txt')))
    for i in range(0, min(sample_num, len(all_files))):
        f = all_files[i]
        print(f)
        file_name = f.split('/')[-1].split('.')[0]  #the file name and remark
        out_f = os.path.join(out_dir,file_name)
        
        if os.path.exists(out_f): 
            print('the file exists. skip....')
            continue
        video_txt = open(f)
        content = video_txt.readlines()
        url = content[0]   #the url file
        try:
            ydl_opts = {'outtmpl': '%(id)s.%(ext)s'}
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                output_file = ydl.prepare_filename(info)
        except:
            print("An exception occurred, maybe because of the downloading limits of youtube.")
            continue
        
        #if video is already downloaded, start extracting frames
        os.makedirs(out_f, exist_ok=True)
        if not os.path.exists(output_file): output_file = output_file.replace('.mp4','.mkv')
        os.rename(output_file, os.path.join(out_f, file_name + '.mp4'))
        line = url
        vidcap = cv2.VideoCapture(os.path.join(out_f, file_name + '.mp4'))
        frame_ind = 1
        frame_file = open(out_f + '/pos.txt','w')
        for num in range(1, len(content), stride):
            line = content[num]
            frame_file.write(line)
            if line == '\n': break
            #line = video_txt.readline()
            ts = line.split(' ')[0][:-3]  #extract the time stamp
            if ts == '': break
            vidcap.set(cv2.CAP_PROP_POS_MSEC,int(ts))      # just cue to 20 sec. position
            success,image = vidcap.read()
            if success:
                cv2.imwrite(out_f + '/' + str(frame_ind) + '.jpg', image)     # save frame as JPEG file
                frame_ind += stride
        frame_file.close()
        video_txt.close()
        
        if remove_video:
            os.remove(os.path.join(out_f, file_name + '.mp4'))

    
def make_dataset_pairs(all_frame_dir, out_dir):
    '''
    #Randomly Extracting Frames with Stride and Resizing (s = 0.5)
    '''
    stride = [10, 20, 30]
    output1 = os.path.join(out_dir, 'target') #target images
    output2 = os.path.join(out_dir, 'source') #source images
    os.makedirs(output1, exist_ok=True)
    os.makedirs(output2, exist_ok=True)

    folder_name = sorted(glob.glob(os.path.join(all_frame_dir, '*')))
    for i in range(len(folder_name)):  #processing the training dataset folder in which contains all the images
        print(i)
        video = folder_name[i]
        file_name = glob.glob(os.path.join(video, '*.jpg'))
        def getint(name):
            num = name.split('/')[-1].split('.')[0]
            return int(num)
        file_name = sorted(file_name, key=getint)
        number_frame = len(file_name)  #how many frames in total
        try:
            for stride_n in stride:  #randomly select three types of strides
                target = np.random.choice(number_frame - stride_n, 1)[0]
                source = target + stride_n
                target_frame = file_name[target]
                source_frame = file_name[source]
                target_frame_image = cv2.imread(target_frame)
                source_frame_image = cv2.imread(source_frame)
                if target_frame_image.shape[0] == 720:  #if smaller size: no need to resize
                    target_frame_image = cv2.resize(target_frame_image, (0,0), fx=0.5, fy=0.5)
                    source_frame_image = cv2.resize(source_frame_image, (0,0), fx=0.5, fy=0.5)
                basename = video.split('/')[-1] + '_' + str(stride_n)
                cv2.imwrite(os.path.join(output1, basename + '_target.png'), target_frame_image)
                cv2.imwrite(os.path.join(output2, basename + '_source.png'), source_frame_image)
        except:
            print('Something wrong with' + video)

if __name__ == "__main__": 
    #using the script to prepare the dataset
    import argparse

    parser = argparse.ArgumentParser(description='Download RealEstate10K Dataset')
    parser.add_argument('--txt_dir', metavar='path', default = './RealEstate10K', required=False,
                        help='path to the original dataset txt files downloaded online')
    parser.add_argument('--frame_dir', metavar='path', default = './RealEstate10K_frames', required=False,
                        help='extract all the frames of videos')
    parser.add_argument('--dataset_dir', metavar='path', default = './RealEstate10K_pair', required=False,
                        help='output paired dataset dir: stride is 10, 20, 30')
    parser.add_argument('--sample_num', type=int, default=1, help='numer of video to download and smple, default 1')
    args = parser.parse_args()

    txt_dir = args.txt_dir
    out_dir = args.frame_dir
    dataset_dir = args.dataset_dir
    download_dataset(txt_dir, out_dir, sample_num=args.sample_num)
    make_dataset_pairs(out_dir, dataset_dir)

