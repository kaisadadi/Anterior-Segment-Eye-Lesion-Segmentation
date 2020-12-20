import json
import numpy as np
import torch
from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)

def print_IoU(data):
    result = {}
    for name in ["IoU", "Dice", "PA"]:
        result[name] = ""
        for idx in range(len(data[name])):
            if idx != len(data[name]) - 1:
                result[name] += "%s:%.3lf," % (str(idx), data[name][idx] / data["cnt"][idx])
            else:
                result[name] += "%s:%.3lf" % (str(idx), data[name][idx] / data["cnt"][idx])
    return result


def IoU_output_function(data, config, *args, **params):
    #write it as you want it to be
    result = {}
    return json.dumps(result, sort_keys=True, ensure_ascii=False)