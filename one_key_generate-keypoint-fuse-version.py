from  mypackage import framedownsample as down
from  mypackage import pair2text as pair
from mypackage import csv2train as dabao
import os
from mypackage import keypoint_fuse as kf
# 使用这个版本的话记得使用media-pipe环境



if __name__ == '__main__':
    print("关键点信息格式, 1. mediapipe输出到原始图像上 2. mediapipe输出到黑色背景图像上 3. mediapipe识别的关键点csv文件 4. how2sign提供的关键点信息" )
    keyinfo= int(input("请输入期望使用的关键点信息格式"))
# 视频分割下采样
    in_folder = input("请输入需要生成帧的文件位置（文件夹）")
    input_folder = in_folder if in_folder else r"F:\how2sign\tete\train"

    cda = down.calculate_dataset_amount(input_folder)
    print(f"视频文件数量为 {cda}")
    frame_folder = input("请输入生成的帧文件存放位置")
    frame_folder = frame_folder if frame_folder else r"F:\Data_preprocessing\dataset\raw"

    dr = input("要进行下采样的倍率，默认为3   ")
    downsample_rate = int(dr if dr else 3)
    down.process_dataset(input_folder, frame_folder, downsample_rate= downsample_rate)
    print(f"\033[91m  帧生成以及下采样结束 Done. \033[0m")

# 选择关键点信息方式

#    input_dir = r"F:\Data_preprocessing\dataset\raw"
    output_dir = input("请输入需要生成关键点图像的文件位置（文件夹）")
    if keyinfo==1:
        kf.mp_draw2raw(input_dir=frame_folder,output_dir=output_dir)
    elif keyinfo==2:
        kf.mp_draw2blk_bg(input_dir=frame_folder,output_dir=output_dir)
    elif keyinfo==3:
        kf.save_keypoints(input_dir=frame_folder,output_dir=output_dir)
    elif keyinfo==4:
        print("记得设置筛选的关键点视频存放位置")
        keypoint_folder = r"E:\how2sign-keypoint-data\openpose_output\video"
        video_path = input_folder

        target_dir = r"F:\Data_preprocessing\dataset\keypoint-video"

        output_dir = r"F:\Data_preprocessing\dataset\keypoint-frame"
        kf.clip_exist_kpvideo(keypoint_folder, video_path, target_dir, output_dir, downsample_rate=10)


# 后面这里要添加一个选择，选择不同的fuse方法 上面一整个部分，第一个是原图形式，对于2，3，4 则要想办法编码然后结合 ，编码的结合也是有着不同的方式的。






# 后续的需要修改


# 跟文本配队
    csv_file = r'.\save\how2sign_realigned_train.csv'
    output_csv = r'F:\Data_preprocessing\output_save\paired_text-frame_data.csv'

    output_data = pair.process_frames_and_text(output_dir, csv_file)
    pair.save_to_csv(output_data, output_csv)
    print(f"\033[91m  帧数据与文本配队结束 Done. \033[0m")
# 添加表头
# 读取CSV文件并解析JSON

    final_df = pair.biaotou(output_csv)
    print(f"\033[91m  表头添加，保存在output_save种，每次重新生成记得删除原文件 Done. \033[0m")
    '''
    这部分留一步给识别关键点并且打印到原始图像上来 融合关键点信息
    
    '''
# 打包压缩为训练数据
    dabao_wenjianming = input("请输出本次生成的训练数据类型， 如labels.train,labels.val  ")
    output_dev_file_path = r'F:\Data_preprocessing\save'
    dabao_baocun_lujing = os.path.join(output_dev_file_path, dabao_wenjianming)

    data_dict = dabao.load_csv_to_dict(output_csv)
    dabao.save_dict_to_gz(data_dict,dabao_baocun_lujing)
    print(f"\033[91m  打包为训练数据格式成功 Done. \033[0m")