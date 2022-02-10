gpu = '0'
random_seed = 0
data_type = 'unseen'
# video_path = 'lip/'
video_path = '/home/max/Desktop/CMLR'
train_list = 'data/data_train.txt'
val_list = 'data/data_val.txt'
# anno_path = 'GRID_align_txt'
anno_path = '/home/max/Desktop/annotation/'

# video_path = '/home/max/Desktop/video'
# train_list = f'data_less/{data_type}_train.txt'
# val_list = f'data_less/{data_type}_val.txt'
# # anno_path = 'GRID_align_txt'
# anno_path = '/home/max/Desktop/grid_corpus/align'

vid_padding = 112
txt_padding = 200
batch_size = 1
base_lr = 0.0003
num_workers = 16
max_epoch = 10000
display = 10
test_step = 1000
save_prefix = f'weights/LipNet_{data_type}'
is_optimize = True

weights = 'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'
