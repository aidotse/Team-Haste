import argparse
import json
import os
import random

import torch
import numpy as np

import ai_haste.train as train
import ai_haste.test as test
import ai_haste.exploratory_analysis as exp_analysis

os.environ["DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTHONHASHSEED"] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="AI-HASTE config file path")
    argparser.add_argument(
        "-c",
        "--conf",
        default="configs/test/config_test_20x.json",
        help="path to configuration file",
    )
    args = argparser.parse_args()
    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_device"]
    # exp_analysis.run(config)
    assert config["run_mode"] in ["train", "test",], "run option does not exist."

    if config["run_mode"] == "train":
        train.run(config)
    elif config["run_mode"] == "test":
        test.run(config)
