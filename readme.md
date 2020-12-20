### README

This is the public code for "Anterior Segment Eye Lesion Segmentation with Advanced Fusion Strategies and Auxiliary Tasks" published in MICCAI-2020.

#### 1. Code Structure

The code is organized as follows:

**config:**

The config folder stores all config files. The config file determines the settings and parameters of model train/eval and test. Each config file has four sections, namely "train", "eval", "data", "model" and "output", which control the settings respectively. The filed in each section is pretty straight-forward, and you can understand the meaning easily. 

I provide a basic version of config file *default.config*, as it determines totally on you to set what you want.

**config_parser:**

The config_reader folder consists of mainly the *ConfigParser* class, which serves to interpret the config files. The logic of config setting is here.

**model:**

The model folder consists of model files. There are three main models as you can see from *model/\__init\__.py*

*Single_Model:*

This model is an interface, and you can use any model implemented in *model/cv/modules*. This model is generally for either eye structure segmentation or eye lesion segmentation.

*Mutli_Task_Model:*

This model is an interface, and you can use any model implemented in *model/cv/modules*. This model is generally for both eye structure segmentation or eye lesion segmentation. You must specify the accuracy function and criterion function carefully, as the final activation on classification can be softmax or sigmoid.

*MyNet:*

This is the model we proposed as described in the original paper. 

**dataset**

The dataset class for loading data.

**formatter**

The formatter before feeding the data into the model, you can take it like the collate_fn in *pytorch dataloader*.

**reader**

Initialize all train/eval/test datasets.

**tools**

The core of the training framework. You need to pay attention to *accuracy_tool.py* (controls how to get accuracy measurement) and *output_tool.py* (controls how to get output info during training or evaluation). 

**utils**

Some self-designed functions, like data splitting.

**train.py & test.py**

The entrance of the codes. Please specify config file, gpu and checkpoint you'd like to load (no means start from beginning).

```python
python3 train.py --config config/xx.config --gpu 0,1,2,3 --checkpoint xxx/xx
```



#### 2. Dataset

As for the dataset, our group decides to open source a much larger dataset than what we claim to release in this paper.  We have decided to release the data along with another short paper with comprehensive experimental results as benchmarks. Once the short paper is published (in few months), we will update the dataset details here.



#### 3. How to contact?

If you have any questions regarding this paper or the code here, please contact **wangke18@mails\.tsinghua.edu.cn**, or **bruuceke@gmail\.com**, as the school email will no-longer be accessible after graduation. 



