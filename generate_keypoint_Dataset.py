from mypackage import framedownsample as down
from mypackage import pair2text as pair
from mypackage import csv2train as dabao
import os
from mypackage import keypoint_fuse as kf

# 使用这个版本的话记得使用media-pipe环境


if __name__ == '__main__':
    print(
        "关键点信息格式, 1. mediapipe输出到原始图像上 2. mediapipe输出到黑色背景图像上 3. mediapipe识别的关键点csv文件 4. how2sign提供的关键点信息")
    keyinfo = int(input("请输入期望使用的关键点信息格式"))
    # 视频分割下采样

    if keyinfo == 1:
        frame_folder = input("请输入帧文件存放位置")
        frame_folder = frame_folder if frame_folder else r"F:\Data_preprocessing\dataset\raw"
        output_dir = input("请输入需要生成关键点图像的文件位置（文件夹）")
        kf.mp_draw2raw(input_dir=frame_folder, output_dir=output_dir)
    elif keyinfo == 2:
        frame_folder = input("请输入帧文件存放位置")
        frame_folder = frame_folder if frame_folder else r"F:\Data_preprocessing\dataset\raw"
        output_dir = input("请输入需要生成关键点图像的文件位置（文件夹）")
        kf.mp_draw2blk_bg(input_dir=frame_folder, output_dir=output_dir)
    elif keyinfo == 3:
        frame_folder = input("请输入帧文件存放位置")
        frame_folder = frame_folder if frame_folder else r"F:\Data_preprocessing\dataset\raw"
        output_dir = input("请输入需要生成关键点图像的文件位置（文件夹）")
        kf.save_keypoints(input_dir=frame_folder, output_dir=output_dir)
    elif keyinfo == 4:
        in_folder = input("请输入raw video位置")
        input_folder = in_folder if in_folder else r"F:\how2sign\tete\train"
        cda = down.calculate_dataset_amount(input_folder)
        video_path = input_folder
        print(f"视频文件数量为 {cda}")
        print("记得设置筛选的关键点视频存放位置")

        keypointfolder = input("输入关键点视频数据存放位置 选项为train,val,test")
        if keypointfolder =="train":
            keypoint_folder = r"E:\how2sign-keypoint-data\openpose_output\video"
        elif keypointfolder=="val":
            keypoint_folder = r"F:\howtosign\val\openpose_output\video"
        elif keypointfolder=="test":
            keypoint_folder = r"F:\howtosign\test\openpose_output\video"
        else :
            print(f"\033[91m  输入错误，请终止程序 Done. \033[0m")
        targetdir = input("输入筛选出来的对应关键点视频存放位置")
        target_dir = targetdir if targetdir else r"F:\Data_preprocessing\dataset\keypoint-video"

        outputdir = input("输出对应关键点的帧数据存放位置")
        output_dir =outputdir if outputdir else r"F:\Data_preprocessing\dataset\keypoint-frame"
        kf.clip_exist_kpvideo(keypoint_folder, video_path, target_dir, output_dir, downsample_rate=5)


