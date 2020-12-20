#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tools.accuracy_tool import IoU_softmax

class Attn_Fusion(nn.Module):
    def __init__(self, n_channels):
        super(Attn_Fusion, self).__init__()
        self.channel_gate = ChannelGate(gate_channels = n_channels,
                                        reduction_ratio = 16,
                                        pool_types = ['avg', 'max']
                                        )
        self.spatial_gate = SpatialGate()
        self.conv = nn.Sequential(
            BasicConv(in_planes = 2 * n_channels,
                      out_planes = n_channels,
                      kernel_size = 1),
            BasicConv(in_planes = n_channels,
                      out_planes = n_channels,
                      kernel_size = 3,
                      padding = 1),
            BasicConv(in_planes = n_channels,
                      out_planes = n_channels,
                      kernel_size = 3,
                      padding = 1),    
        )
        self.conv_1_1 = nn.Conv2d(in_channels = 2 * n_channels,
                                  out_channels = n_channels,
                                  kernel_size = 1,
                                  )

    def forward(self, lesion_feature, structure_feature):
        fusion_feature = torch.cat([lesion_feature, structure_feature], dim=1)  #channel-wise concate
        fusion_feature = self.conv(fusion_feature)

        channel_out = fusion_feature * self.channel_gate(lesion_feature)
        spatial_out = fusion_feature * self.spatial_gate(lesion_feature)

        out_feature = torch.cat([channel_out, spatial_out], dim=1)
        out_feature = self.conv_1_1(out_feature)
        return out_feature


class Lesion_Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Lesion_Net, self).__init__()
        self.resnet = [*list(models.resnet50(pretrained=True).children())][:-3]
        self.layer_1 = nn.Sequential(
            self.resnet[0],
            self.resnet[1],
            self.resnet[2],
            self.resnet[3]
        )  # 1/4 1/4
        self.layer_2 = self.resnet[4]  # 1/4 1/4 exit 1
        self.layer_3 = self.resnet[5]  # 1/8 1/8 exit 2
        self.layer_4 = self.resnet[6]  # 1/16 1/16 exit 3

        # dilated conv to make up for downsample
        self.layer_5 = make_layer(Bottleneck, in_channels=4 * 256, channels=512, num_blocks=3, stride=1, dilation=2)

        # ASPP layer
        self.ASPP = ASPP_Bottleneck(num_classes=n_classes, structure=False)

        #Attention Fusion layer
        self.fusion_1 = Attn_Fusion(n_channels = 256)
        self.fusion_2 = Attn_Fusion(n_channels = 512)
        self.fusion_3 = Attn_Fusion(n_channels = 1024)

        #upsampling layer
        self.conv_1_1_1024 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv_1_1_512 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=1)
        self.final_conv = nn.Sequential(
            BasicConv(in_planes = 512 + 256 + 256, out_planes = 256, kernel_size = 3, padding=1),
            BasicConv(in_planes = 256, out_planes = 256, kernel_size=3, padding=1),
            BasicConv(in_planes = 256, out_planes = n_classes, kernel_size=1),
        )

        #auxiliary task feature gen
        self.conv_feature = nn.Sequential(
            BasicConv(in_planes = 2048, out_planes = 512, kernel_size = 3, stride = 2, padding = 1),
            BasicConv(in_planes = 512, out_planes = 256, kernel_size = 3, stride = 2, padding = 1),
            nn.AdaptiveAvgPool2d(1)
        )


    def forward(self, x, out1, out2, out3):
        down_1 = self.layer_1(x)
        down_2 = self.layer_2(down_1)
        down_3 = self.layer_3(down_2 + self.fusion_1(down_2, out1))
        down_4 = self.layer_4(down_3 + self.fusion_2(down_3, out2))
        down_5 = self.layer_5(down_4 + self.fusion_3(down_4, out3))

        #ASPP module
        feature = self.ASPP(down_5)

        #out feature vec
        out_feature_vec = self.conv_feature(down_5).squeeze(3).squeeze(2)

        #upsample module
        size = (32, 32)
        up_1 = F.interpolate(feature, (64, 64), mode='bilinear', align_corners=True)
        conv_fuse_1 = self.conv_1_1_1024(down_3)
        up_1 = torch.cat([up_1, conv_fuse_1], dim=1)
        up_2 = F.interpolate(up_1, (128, 128), mode='bilinear', align_corners=True)
        conv_fuse_2 = self.conv_1_1_512(down_2)
        up_3 = torch.cat([up_2, conv_fuse_2], dim=1)
        up_3 = self.final_conv(up_3)
        out = F.interpolate(up_3, (512, 512), mode='bilinear', align_corners=True)

        return out, out_feature_vec


class Structure_Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Structure_Net, self).__init__()

        #resnet layers
        self.resnet = [*list(models.resnet50(pretrained=True).children())][:-3]
        self.conv = self.resnet[0]
        self.layer_1 = nn.Sequential(
                self.resnet[0],
                self.resnet[1],
                self.resnet[2],
                self.resnet[3]
        )  # 1/4 1/4
        self.layer_2 = self.resnet[4]  #1/4 1/4 exit 1
        self.layer_3 = self.resnet[5]  #1/8 1/8 exit 2
        self.layer_4 = self.resnet[6]  #1/16 1/16 exit 3

        #dilated conv to make up for downsample
        self.layer_5 = make_layer(Bottleneck, in_channels=4 * 256, channels=512, num_blocks=3, stride=1, dilation=2)

        #ASPP layer
        self.ASPP = ASPP_Bottleneck(num_classes=n_classes, structure=True)

        # auxiliary task feature gen
        self.conv_feature = nn.Sequential(
            BasicConv(in_planes=2048, out_planes=512, kernel_size=3, stride=2, padding=1),
            BasicConv(in_planes=512, out_planes=256, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        out0 = self.layer_1(x)
        out1 = self.layer_2(out0) #1/4 1/4
        out2 = self.layer_3(out1) #1/8 1/8
        out3 = self.layer_4(out2)  #1/16 1/16
        feature_map = self.layer_5(out3)

        # out feature vec
        out_feature_vec = self.conv_feature(feature_map).squeeze(3).squeeze(2)

        output = self.ASPP(feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

        return output, (out1, out2, out3), out_feature_vec


def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = F.relu(self.bn2(self.conv2(out))) 
        out = self.bn3(self.conv3(out)) 

        out = out + self.downsample(x) 

        out = F.relu(out) 

        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    #channel gate attention
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) 
        return scale


class ASPP_Bottleneck(nn.Module):
    def __init__(self, num_classes, structure=True):
        super(ASPP_Bottleneck, self).__init__()

        self.structure = structure

        self.conv_1x1_1 = nn.Conv2d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        if structure:
            self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)  # (1280 = 5*256)
            self.bn_conv_1x1_3 = nn.BatchNorm2d(256)
            self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)
        else:
            self.conv_1x1_3 = nn.Conv2d(1280, 512, kernel_size=1) # (1280 = 5*256)
            self.bn_conv_1x1_3 = nn.BatchNorm2d(512)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2] 
        feature_map_w = feature_map.size()[3] 

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) 
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) 
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) 
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) 

        out_img = self.avg_pool(feature_map) 
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) 
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") 

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) 

        if self.structure == False:
            return out
        out = self.conv_1x1_4(out) 
        return out


class MFB(nn.Module):
    def __init__(self, config):
        super(MFB, self).__init__()
        self.k = config.getint("k")
        self.MFB_in_dim = 256
        self.MFB_out_dim = config.getint("MFB_dim")
        
        self.fc_lesion = nn.Linear(self.MFB_in_dim, self.k * self.MFB_out_dim)
        self.fc_structure = nn.Linear(self.MFB_in_dim, self.k * self.MFB_out_dim)
    
    def forward(self, lesion_feature, structure_feature):
        out = self.fc_lesion(lesion_feature) * self.fc_structure(structure_feature)
        out = out.view(-1, self.MFB_out_dim, self.k)
        out = torch.sum(out, dim = 2, keepdim = False)
        return out


class Auxiliary_task(nn.Module):
    def __init__(self, config):
        super(Auxiliary_task, self).__init__()
        self.use_MFB = config.getboolean("model", "use_MFB")
        self.MFB_out_dim = config.getint("MFB_dim")
        if self.use_MFB:
            in_dim = self.MFB_out_dim
            self.MFB_fuse = MFB(config)
        else:
            in_dim = 2 * 256 
        self.task1 = nn.Sequential(
            nn.Linear(in_dim, int(in_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(in_dim / 2), 4),
            nn.Sigmoid()
        )
        self.task2 = nn.Sequential(
            nn.Linear(in_dim, int(in_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(in_dim / 2), 4),
            nn.Sigmoid()
        )

    def forward(self, lesion_feature, structure_feature):
        if self.use_MFB:
            fuse_feature = self.MFB_fuse(lesion_feature, structure_feature)
        else:
            fuse_feature = torch.cat([lesion_feature, structure_feature], dim=1)
        task_1_output = self.task1(fuse_feature)
        task_2_output = self.task2(fuse_feature)
        return task_1_output, task_2_output


class MyNet(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(MyNet, self).__init__()
        self.input_channel_num = config.getint("model", "input_channel_num")
        self.output_class_num = config.getint("model", "output_class_num")
        self.lambda1 = config.getfloat("model", "lambda1")
        self.lambda2 = config.getfloat("model", "lambda2")
        assert self.output_class_num == 9
        self.lesion_net = Lesion_Net(self.input_channel_num, n_classes=5)
        self.structure_net = Structure_Net(self.input_channel_num, n_classes=4)
        self.auxiliary_task = Auxiliary_task(config)

        self.criterion_lesion = nn.CrossEntropyLoss(reduction="mean", weight=torch.Tensor([1, 25, 11, 18, 320]))
        self.criterion_structure = nn.CrossEntropyLoss(reduction="mean")
        self.criterion_auxiliary = nn.BCELoss(reduction="mean")
        self.accuracy_function = IoU_softmax

    def forward(self, data, config, gpu_list, acc_result, mode):
        assert len(acc_result) == 2
        x = data['input']
        structure_logits, (out1, out2, out3), structure_feature = self.structure_net(x)
        lesion_logits, lesion_feature = self.lesion_net(x, out1, out2, out3)

        task_1_output, task_2_output = self.auxiliary_task(lesion_feature, structure_feature)
        loss_task_1 = self.criterion_auxiliary(task_1_output, data["label_task1"])
        loss_task_2 = self.criterion_auxiliary(task_2_output, data["label_task2"])

        structure_prediction = torch.argmax(structure_logits, dim=1, keepdim=False)  
        lesion_prediction = torch.argmax(lesion_logits, dim=1, keepdim=False) 
        prediction = torch.cat([structure_prediction, lesion_prediction], dim=1).detach()

        if "label_lesion" in data.keys():
            lesion_label = data["label_lesion"].long()
            structure_label = data["label_structure"].long()
            loss_lesion = self.criterion_lesion(lesion_logits, lesion_label)
            loss_structure = self.criterion_structure(structure_logits, structure_label)
            structure_acc_result = self.accuracy_function(structure_prediction, structure_label,
                                                          result=acc_result[0],
                                                          output_class=4)
            lesion_acc_result = self.accuracy_function(lesion_prediction, lesion_label,
                                                       result=acc_result[1],
                                                       output_class=5)
            return {"loss": loss_lesion + loss_structure + self.lambda1 * loss_task_1 + self.lambda2 * loss_task_2,
                    "acc_result": [structure_acc_result, lesion_acc_result],
                    "prediction": prediction,
                    "file": data["file"]}

        return {"prediction": prediction, "file": data["file"]}


if __name__ == "__main__":
    pass