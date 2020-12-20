import logging
import os
import torch
from torch.autograd import Variable
import json
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from tools.output_tool import print_IoU

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)


def valid(model, dataset, epoch, writer, config, gpu_list, output_function):
    model.eval()

    Multi_task = False
    if config.get("model", "model_name") in ["Multi", "MyNet"]:
        Multi_task = True

    acc_result = None
    if Multi_task:
        acc_result = [None, None]
    output_file = config.get("output", "output_file") if config.get("output", "output_file") != "None" else None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1
    more = ""
    if total_len < 10000:
        more = "\t"

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, acc_result, "valid")

        loss, acc_result = results["loss"], results["acc_result"]
        total_loss += float(loss)
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(epoch, "valid", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)



    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    if Multi_task:
        output_info = output_function(acc_result[1], config, type="lesion") + \
                      output_function(acc_result[0], config, type="structure")
    else:
        output_info = output_function(acc_result, config)
    output_value(epoch, "valid", "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    if output_file is not None:
        f = open(output_file, "a")
        f.write("valid: " + json.dumps(print_IoU(acc_result)) + "\n")
        f.close()


    model.train()
