import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.accuracy_init import init_accuracy_function
from tools.accuracy_tool import IoU_softmax, IoU_sigmoid

from model.cv.modules.Unet import pre_UNet
from model.cv.modules.Unet_zoos import UNet_raw, UNet_Nested_dilated
from model.cv.modules.deeplab.deeplab import DeepLabV3
from model.cv.modules.FCN.fcn8s import FCN8s
from model.cv.modules.UNET_plus import UNet_plus
from model.cv.modules.ladderNet import LadderNetv6
from model.cv.modules.segnet import SegNet
from model.cv.modules.PspNet import PSPDenseNet, PSPNet
from model.cv.modules.PsPNet_res import PSPNet_Res

from model.loss import BCEWithLogitsLoss2d


module_list = {"UNet": pre_UNet,
              "UNet_raw": UNet_raw,
              "Dilated_UNet": UNet_Nested_dilated,
              "FCN8s": FCN8s,
              "deeplabv3": DeepLabV3,
               "UNet++": UNet_plus,
               "LadderNet": LadderNetv6,
               "SegNet": SegNet,
               "PspNet": PSPDenseNet,
               "PspNet_res": PSPNet
              }


class Single_Model(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Single_Model, self).__init__()
        self.input_channel_num = config.getint("model", "input_channel_num")
        self.output_class_num = config.getint("model", "output_class_num")
        self.label_weight = [int(val) for val in config.get("model", "label_weight").split(",")]

        self.net = module_list[config.get("model", "module_name")](n_channels = self.input_channel_num,
                                                                   n_classes = self.output_class_num)

        self.criterion = [nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.Tensor([weight]).cuda()) for weight in self.label_weight]
        self.accuracy_function = IoU_sigmoid

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['input']
        logits = self.net(x)   
        if logits.shape[3] != 512:
            logits = nn.Upsample((512, 512), mode='bilinear')(logits)  
        prediction = []
        for idx in range(len(self.label_weight)):
            prediction.append(torch.ge(nn.Sigmoid()(logits[:, idx, :, :]), 0.5))
        prediction = torch.stack(prediction, axis=1).cuda()
        if "label" in data.keys():
            label = data["label"]
            loss = 0
            for idx in range(len(self.label_weight)):
                loss += self.criterion[idx](logits[:, idx, :, :], label[:, idx, :, :])
            acc_result = self.accuracy_function(prediction, label, result=acc_result, output_class=self.output_class_num)
            return {"loss": loss, "acc_result": acc_result, "prediction": prediction, "file": data["file"]}

        return {"prediction": prediction, "file": data["file"]}



