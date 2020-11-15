import json
import os

import ai_haste.loss as loss
import ai_haste.model as model

from .data import main as data
from .tester import main as tester
from .trainer import main as trainer


def run(config):
    test_dataloader = data.run(config["data"], config["run_mode"])
    exp_folder = config["exp_folder"]
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    magnification = config["data"]["magnification"]
    
    output_save_folder = os.path.join(exp_folder, f"{magnification}_images")
    if not os.path.exists(output_save_folder):
        os.makedirs(output_save_folder)
    
    models = []
    for model_config in config["models"]:
        models.append(
            {
                "model_path": model_config["path"],
                "model": getattr(model, model_config["type"])(**model_config["args"]),
            }
        )

    tester.run(config["test"], test_dataloader, models, output_save_folder)

