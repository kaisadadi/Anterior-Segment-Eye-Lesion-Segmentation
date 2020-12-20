import torch
import numpy as np

from formatter.Basic import BasicFormatter


class Single_Model_Formatter(BasicFormatter):
    #single_task model, supervised
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.config = config
        self.mode = mode
        self.normalization = True

    def process(self, data, config, mode, *args, **params):
        out_data = []
        out_label = []
        out_file = []
        for item in data:
            out_data.append(item["data"].astype(np.float32) / 255.0)
            if "label" in item.keys():
                out_label.append(item["label"].astype(np.float32))
            out_file.append(item["file"])
        out_data = torch.from_numpy(np.array(out_data))
        out_data = out_data.permute(0, 3, 1, 2)
        out_label = torch.from_numpy(np.array(out_label))
        return {"input": out_data,
                "label": out_label,
                "file": out_file}
