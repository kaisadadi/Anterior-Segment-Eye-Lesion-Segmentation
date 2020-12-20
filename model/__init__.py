from .cv.single_model import Single_Model
from .cv.multi_task_model import Multi_Task_Model
from .cv.MyNet import MyNet

model_list = {
    "Single": Single_Model,
    "Multi": Multi_Task_Model,
    "MyNet": MyNet
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
