import argparse
import os
import torch
import yaml
from data.argparser import args_parsing
import numpy as np
from datetime import datetime
from experiment.exp import Exp_XGBoost

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

args = args_parsing()

# # DEBUGGING ONLY
# ####
# args.loss = 'linex'

# ####

exp = Exp_XGBoost(args)

exp.train()

exp.vali()

print(' --- ')