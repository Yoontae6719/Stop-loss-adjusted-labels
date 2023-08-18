import argparse
import os
import numpy as np
import random
from datetime import datetime
from modeling import my_model

fix_seed = 2022
random.seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Stock Prediction')


# data loader
parser.add_argument('--data', type=str,  default='CL', help='dataset')
parser.add_argument('--time', type=str,  default='4h', help='dataset time')
parser.add_argument('--target_col', type=str,  default='Target_Label', help='taget columns')

parser.add_argument('--date', type = int, default = 1, help = "using time data")
parser.add_argument('--norm', type=int,  default=0, help='technical indicators normalization')

parser.add_argument('--model_weight_path', type=str,  default='model_weight', help='dataset') #model_weight_wo_time
parser.add_argument('--predict_result_path', type=str,  default='predict_result', help='dataset') #predict_result_wo_time



args = parser.parse_args()

my_model(args)