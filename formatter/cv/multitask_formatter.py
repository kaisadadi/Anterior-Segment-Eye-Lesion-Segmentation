import torch
import numpy as np

from formatter.Basic import BasicFormatter


class Multitask_Formatter(BasicFormatter):
    #multi_task model, supervised
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.config = config
        self.mode = mode
        self.normalization = True

    def transfer_label(self, label, num_class):
        sigmoid_label = np.zeros([num_class, 512, 512], np.float32)
        for idx in range(num_class):
            sigmoid_label[idx] = (label == idx)
        return sigmoid_label

    def get_task1_label(self, label):
        task1_label = []
        for idx in range(1, 5):
            sum = np.sum((label[1] == idx))
            if sum > 100:
                task1_label.append(1)
            else:
                task1_label.append(0)
        return task1_label

    def get_task2_label(self, label):
        task2_label = []
        lesion_mask = (label[1] > 0).astype(np.int32)
        for idx in range(4):
            structure = (label[0] == idx).astype(np.int32)
            if np.sum(lesion_mask * structure) > 100:  
                task2_label.append(1)
            else:
                task2_label.append(0)
        return task2_label


    def process(self, data, config, mode, *args, **params):
        out_data = []
        out_label_lesion = []
        task1_label = []
        task2_label = []
        out_label_structure = []
        out_file = []
        for item in data:
            out_data.append(item["data"].astype(np.float32) / 255.0)
            if "label" in item.keys():
                assert item["label"].shape[0] == 5
                out_label_lesion.append(item["label"][1:].astype(np.float32))
                out_label_structure.append(item["label"][0].astype(np.float32))
                task1_label.append(self.get_task1_label(item["label"]))
                task2_label.append(self.get_task2_label(item["label"]))
            out_file.append(item["file"])
        out_data = torch.from_numpy(np.array(out_data))
        out_data = out_data.permute(0, 3, 1, 2)
        out_label_lesion = torch.from_numpy(np.array(out_label_lesion))
        out_label_structure = torch.from_numpy(np.array(out_label_structure)).squeeze(1)
        task1_label = torch.from_numpy(np.array(task1_label)).float()
        task2_label = torch.from_numpy(np.array(task2_label)).float()
        return {"input": out_data,
                "label_lesion": out_label_lesion,
                "label_structure": out_label_structure,
                "label_task1": task1_label,
                "label_task2": task2_label,
                "file": out_file}
