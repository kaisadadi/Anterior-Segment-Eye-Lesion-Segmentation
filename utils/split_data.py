import os
import random
import numpy as np
import shutil

class config:
    input_file = r"__YOUT_PATH__/data"
    label_file = r"__YOUT_PATH__/label"
    output_train_data_file = r"__YOUT_PATH__/train_data"
    output_train_label_file = r"__YOUT_PATH__/train_label_0"
    output_valid_data_file = r"__YOUT_PATH__/valid_data"
    output_valid_label_file = r"__YOUT_PATH__/valid_label"
    ratio = 0.8 

if __name__ == "__main__":
    opt = config()
    for out_path in [opt.output_train_data_file,
                     opt.output_train_label_file,
                     opt.output_valid_data_file,
                     opt.output_valid_label_file]:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.mkdir(out_path)
    file_list = os.listdir(opt.input_file)
    id_list = []
    for file in file_list:
        id_list.append(int(file[:-4]))
    random.shuffle(id_list)
    train_id = id_list[:int(len(id_list) * opt.ratio)]
    valid_id = id_list[int(len(id_list) * opt.ratio):]
    for file in file_list:
        data_file = os.path.join(opt.input_file, file)
        label_file = os.path.join(opt.label_file, file)
        if int(file[:-4]) in train_id:
            shutil.copy(data_file, os.path.join(opt.output_train_data_file, file))
            shutil.copy(label_file, os.path.join(opt.output_train_label_file, file))
        else:
            shutil.copy(data_file, os.path.join(opt.output_valid_data_file, file))
            shutil.copy(label_file, os.path.join(opt.output_valid_label_file, file))