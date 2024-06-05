from  mypackage import framedownsample as down
from  mypackage import pair2text as pair
from mypackage import csv2train as dabao
import os
if __name__ == '__main__':
# 视频分割下采样
    in_folder = input("请输入需要生成帧的文件位置（文件夹）")
    input_folder = in_folder if in_folder else r"F:\Data_preprocessing\Test_liucheng\raw_video"

    cda = down.calculate_dataset_amount(input_folder)
    print(f"视频文件数量为 {cda}")
    frame_folder = input("请输入生成的帧文件存放位置")
    frame_folder = frame_folder if frame_folder else r"F:\Data_preprocessing\output_save\frame_tem_save"

    dr = input("要进行下采样的倍率，默认为3   ")
    downsample_rate = int(dr if dr else 3)
    down.process_dataset(input_folder, frame_folder, downsample_rate= downsample_rate)
    print(f"\033[91m  帧生成以及下采样结束 Done. \033[0m")

# 跟文本配队
    csv_file = r'.\save\how2sign_realigned_train.csv'
    output_csv = r'F:\Data_preprocessing\output_save\paired_text-frame_data.csv'

    output_data = pair.process_frames_and_text(frame_folder, csv_file)
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