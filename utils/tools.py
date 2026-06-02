import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from ptflops import get_model_complexity_info
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 16

plt.switch_backend('agg')

def get_max_gpu_memory():

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return max(allocated, reserved)

def track_max_memory():

    if 'max_memory' not in track_max_memory.__dict__:
        track_max_memory.max_memory = 0

    current_memory = get_max_gpu_memory()

    if current_memory > track_max_memory.max_memory:
        track_max_memory.max_memory = current_memory

    return track_max_memory.max_memory


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth',color='blue', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction',color='orange', linewidth=2)
    plt.legend(loc='upper left',fontsize=11)
    plt.savefig(name, bbox_inches='tight')


def calculate_inference_time(model, input_tensor, device):
    model.eval()

    start_time = time.time()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)

    end_time = time.time()
    inference_time = end_time - start_time
    return inference_time

def test_params_flop(model, x_shape):
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))

    macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
