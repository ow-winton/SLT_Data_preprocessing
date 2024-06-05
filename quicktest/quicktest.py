from mypackage import keypoint_fuse as kf
# input_dir = r"F:\Data_preprocessing\dataset\raw"
# output_dir = r"F:\Data_preprocessing\dataset\media-pipe-point-info"
# kf.mp_draw2raw(input_dir=input_dir,output_dir=output_dir)

# kf.mp_draw2blk_bg(input_dir=input_dir,output_dir=output_dir)

# kf.save_keypoints(input_dir=input_dir,output_dir=output_dir)

keypoint_folder = r"E:\how2sign-keypoint-data\openpose_output\video"
video_path = r"F:\how2sign\tete\train"
target_dir = r"F:\Data_preprocessing\dataset\keypoint-video"
output_dir = r"F:\Data_preprocessing\dataset\keypoint-frame"
kf.clip_exist_kpvideo(keypoint_folder, video_path, target_dir, output_dir,downsample_rate=10)