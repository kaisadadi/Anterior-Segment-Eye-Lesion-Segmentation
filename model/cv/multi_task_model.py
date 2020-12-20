#coding=utf-8
#for multitask model, 3.16 by wk
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

from model.loss import BCEWithLogitsLoss2d


module_list = {"UNet": pre_UNet,
              "UNet_raw": UNet_raw,
              "Dilated_UNet": UNet_Nested_dilated,
              "FCN8s": FCN8s,
              "deeplabv3": DeepLabV3,
               "UNet++": UNet_plus,
              }


class Multi_Task_Model(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Multi_Task_Model, self).__init__()
        self.input_channel_num = config.getint("model", "input_channel_num")
        self.output_class_num = config.getint("model", "output_class_num")
        assert self.output_class_num == 4 + 4

        self.net = module_list[config.get("model", "module_name")](n_channels = self.input_channel_num,
                                                                   n_classes = self.output_class_num)

        self.criterion_lesion = BCEWithLogitsLoss2d()
        self.criterion_structure = nn.CrossEntropyLoss(reduction="mean")
        self.accuracy_function_structure = IoU_softmax
        self.accuracy_function_lesion = IoU_sigmoid

    def forward(self, data, config, gpu_list, acc_result, mode):
        assert len(acc_result) == 2
        x = data['input']
        logits = self.net(x)   
        if logits.shape[3] != 512:
            logits = nn.Upsample((512, 512), mode='bilinear')(logits)  
        structure_logits = logits[:, :4, :, :]
        lesion_logits = nn.Sigmoid()(logits[:, 4:, :, :])   

        structure_prediction = torch.argmax(structure_logits, dim = 1, keepdim=True)  
        lesion_prediction = torch.ge(lesion_logits, 0.5).long()  
        prediction = torch.cat([structure_prediction, lesion_prediction], dim = 1).detach()
        if "label_lesion" in data.keys():
            lesion_label = data["label_lesion"].float()
            structure_label = data["label_structure"].long()
            loss_lesion = self.criterion_lesion(lesion_logits, lesion_label)
            loss_structure = self.criterion_structure(structure_logits, structure_label)
            structure_acc_result = self.accuracy_function_structure(structure_prediction, structure_label,
                                                                    result=acc_result[0],
                                                                    output_class=4)
            lesion_acc_result = self.accuracy_function_lesion(lesion_prediction, lesion_label,
                                                                result=acc_result[1],
                                                                output_class=4)
            return {"loss": loss_lesion + loss_structure,
                    "acc_result": [structure_acc_result, lesion_acc_result],
                    "prediction": prediction,
                    "file": data["file"]}

        return {"prediction": prediction, "file": data["file"]}



