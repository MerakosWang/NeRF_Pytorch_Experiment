# -*- coding:utf8 -*-
import cv2
import os
import shutil


def get_frame_from_video(video_name, interval):
    """
    Args:
        video_name:输入视频名字
        interval: 保存图片的帧率间隔
    Returns:
    """

    # 保存图片的路径
    save_path = video_name.split('.mp4')[0] + '/'
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)

    # 开始读视频
    video_capture = cv2.VideoCapture(video_name)
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        i += 1
        if i % interval == 0:
            # 保存图片
            j += 1
            save_name = save_path + str(j) + '.jpg'
            cv2.imwrite(save_name, frame)
            print('image of %s is saved' % save_name)
        if not success:
            print('video is all read')
            break



if __name__ == '__main__':
    video_name = './Trim4.mp4'  # 视频文件名字
    interval = 5
    get_frame_from_video(video_name, interval)

