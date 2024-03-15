# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Logging & tensorboard utils
"""

import os
import warnings
from pathlib import Path


import tensorflow as tf
from utils.tf_general import LOGGER, colorstr
import cv2
import wandb
# from utils.loggers.clearml.clearml_utils import ClearmlLogger
# from utils.loggers.wandb.wandb_utils import WandbLogger
# from utils.plots import plot_images, plot_labels, plot_results
# from utils.torch_utils import de_parallel

LOGGERS = ('csv', 'tb', 'wandb', 'clearml', 'comet')  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv('RANK', -1))

#

class GenericLogger:
    """
    YOLOv5 General purpose logger for non-task specific logging
    Usage: from utils.loggers import GenericLogger; logger = GenericLogger(...)
    Arguments
        opt:             Run arguments
        console_logger:  Console logger
        include:         loggers to include
    """

    def __init__(self, opt, console_logger, res_table_cols, include=('tb', 'wandb')):
        # init default loggers
        self.save_dir = Path(opt.save_dir)
        self.include = include
        self.console_logger = console_logger
        self.csv = self.save_dir / 'results.csv'  # CSV logger

        self.res_table_cols=['train/epoch'] + res_table_cols
        import wandb
        self.res_table = wandb.Table(columns=self.res_table_cols)

        # self.csv = self.save_dir / 'results.csv'  # CSV logger
        if 'tb' in self.include:
            prefix = colorstr('TensorBoard: ')
            self.console_logger.info(
                f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/")
            self.test_summary_writer = tf.summary.create_file_writer(str(self.save_dir))


    def log_metrics(self, metrics, epoch):
        if self.csv:
            keys, vals = list(metrics.keys()), list(metrics.values())
            n = len(metrics) + 1  # number of cols
            s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
            with open(self.csv, 'a') as f:
                f.write(s + ('%23.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')
        self.res_table.add_data(*([epoch] +list(metrics.values())))
        # workaround wandb bug: log runs once after creating a table instance. So recreate an instance after add data:
        self.res_table = wandb.Table(data=self.res_table.data, columns=self.res_table_cols)
        wandb.log({"results-table": self.res_table})


    def log_images(self, files, name='Images', epoch=0):
        # Log images to all loggers
        files = [Path(f) for f in (files if isinstance(files, (tuple, list)) else [files])]  # to Path
        files = [f for f in files if f.exists()]  # filter by exists

        # if self.tb:
        for f in files:
                # self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')
            tf.summary.image(f.stem, cv2.imread(str(f))[..., ::-1], epoch)

        # if self.wandb:
        # wandb.log({name: [wandb.Image(str(f), caption=f.name) for f in files]}, step=epoch)

    def log_graph(self, model, imgsz=(640, 640)):
        pass
        # Log model graph to all loggers
        # if self.tb:
        #     log_tensorboard_graph(self.tb, model, imgsz)

    def log_model(self, model_path, epoch=0, metadata={}):
        pass
        # Log model to all loggers
        # if self.wandb:
        #     art = wandb.Artifact(name=f'run_{wandb.run.id}_model', type='model', metadata=metadata)
        #     art.add_file(str(model_path))
        #     wandb.log_artifact(art)

    def update_params(self, params):
        pass
        # Update the parameters logged
        # if self.wandb:
        #     wandb.run.config.update(params, allow_val_change=True)



def web_project_name(project):
    # Convert local project name to web project name
    if not project.startswith('runs/train'):
        return project
    suffix = '-Classify' if project.endswith('-cls') else '-Segment' if project.endswith('-seg') else ''
    return f'YOLOv5{suffix}'
