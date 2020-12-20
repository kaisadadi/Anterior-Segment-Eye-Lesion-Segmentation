import logging
import os
import torch
from torch.autograd import Variable
from timeit import default_timer as timer
import cv2
import numpy as np

from tools.eval_tool import gen_time_str, output_value

logger = logging.getLogger(__name__)


def test(parameters, config, gpu_list):
    model = parameters["model"]
    dataset = parameters["test_dataset"]
    output_class_num = config.getint("model", "output_class_num")
    model_name = config.get("output", "model_name")
    output_path = os.path.join(config.get("output", "test_output_path"), model_name)  #output image
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    model.eval()

    Multi_task = False
    if config.get("model", "model_name") in ["Multi", "MyNet"]:
        Multi_task = True
    acc_result = None
    if Multi_task:
        acc_result = [None, None]
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = "testing"

    output_time = config.getint("output", "output_time")
    step = -1
    result = []

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, acc_result, "test")
        cnt += 1

        prediction = results["prediction"].detach().cpu().numpy()  # batch * width * height
        file = results["file"]
        out_data_path = config.get("output", "output_data_path")
        assert len(prediction.shape) == 3
        for batch_idx in range(prediction.shape[0]):
            np.save(os.path.join(out_data_path, file[batch_idx]), prediction[batch_idx])
        print("step = %d\n" % step)

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    return result
