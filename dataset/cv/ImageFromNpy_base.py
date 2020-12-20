import json
from torch.utils.data import Dataset
import cv2
import os
import numpy as np


class ImageFromNpyDataset_V1(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.label_path = config.get("data", "%s_label_path" % mode)

        self.target_label = [int(val) for val in config.get("data", "target_label").split(",")]

        self.load_mem = config.getboolean("data", "load_into_mem")
        self.file_list = os.listdir(self.data_path)
        self.data_list = [None] * len(self.file_list)
        self.file_record = [0] * len(self.file_list)

        if self.load_mem:
            for file in self.file_list:
                data_file = os.path.join(self.data_path, file)
                label_file = os.path.join(self.label_path, file)
                self.data_list.append({"data": np.load(data_file), "label": self.process_label(label_file), "file": file})

    def process_label(self, label_file):
        raw_label = np.load(label_file)
        real_label = raw_label[self.target_label, :, :]
        if len(real_label.shape) == 2:
            real_label = real_label[np.newaxis, :, :]
        return real_label

    def __getitem__(self, item):
        if self.load_mem:
            return {
                "data": self.data_list[item]["data"],
                "label": self.data_list[item]["label"],
                "file": self.data_list[item]["file"]
            }
        else:
            if self.file_record[item] == 0:
                self.file_record[item] = 1
                data_file = os.path.join(self.data_path, self.file_list[item])
                label_file = os.path.join(self.label_path, self.file_list[item])
                self.data_list[item] = {"data": np.load(data_file), "label": self.process_label(label_file), "file": self.file_list[item]}
            return {
                "data": self.data_list[item]["data"],
                "label": self.data_list[item]["label"],
                "file": self.data_list[item]["file"]
            }

    def __len__(self):
        return len(self.file_list)
