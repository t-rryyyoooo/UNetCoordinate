from importlib import import_module
import os
import pytorch_lightning as pl
import json
import argparse
from pathlib import Path
import sys

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", help="/home/vmlab/Desktop/data/patch/Abdomen/28-44-44/image")
    parser.add_argument("model_savepath", help="/home/vmlab/Desktop/data/modelweight/Abdomen/28-44-44/mask")
    parser.add_argument("module_name", help="Model directory name under model/.")
    parser.add_argument("system_name", help="The class name in system.py")
    parser.add_argument("checkpoint_name", help="Checkpoint class name.")
    parser.add_argument("--train_list", help="00 01", nargs="*", default= "00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19")
    parser.add_argument("--val_list", help="20 21", nargs="*", default="20 21 22 23 24 25 26 27 28 29")
    parser.add_argument("--log", help="/home/vmlab/Desktop/data/log/Abdomen/28-44-44/mask", default="log")
    parser.add_argument("--in_channel", help="Input channlel", type=int, default=1)
    parser.add_argument("--num_class", help="The number of classes.", type=int, default=14)
    parser.add_argument("--lr", help="Default 0.001", type=float, default=0.001)
    parser.add_argument("--batch_size", help="Default 6", type=int, default=6)
    parser.add_argument("--dropout", help="Default 6", type=float, default=0.5)
    parser.add_argument("--num_workers", help="Default 6.", type=int, default=6)
    parser.add_argument("--epoch", help="Default 50.", type=int, default=50)
    parser.add_argument("--gpu_ids", help="Default 0.", type=int, default=0, nargs="*")

    parser.add_argument("--api_key", help="Your comet.ml API key.")
    parser.add_argument("--project_name", help="Project name log is saved.")
    parser.add_argument("--experiment_name", help="Experiment name.", default="3DU-Net")


    args = parser.parse_args()

    return args

def main(args):
    criteria = {
            "train" : args.train_list, 
            "val" : args.val_list
            }

    sys.path.append("..")
    system_path = "." + args.module_name + ".system"
    checkpoint_path = "." + args.module_name + ".modelCheckpoint"
    system_module = import_module(system_path, "model")
    checkpoint_module = import_module(checkpoint_path, "model")
    UNetSystem = getattr(system_module, args.system_name)
    checkpoint = getattr(checkpoint_module, args.checkpoint_name)
    system = UNetSystem(
            dataset_path = args.dataset_path,
            criteria = criteria,
            in_channel = args.in_channel,
            num_class = args.num_class,
            learning_rate = args.lr,
            batch_size = args.batch_size,
            dropout = args.dropout,
            num_workers = args.num_workers, 
            checkpoint = checkpoint(args.model_savepath)
            )

    if args.api_key != "No": 
        from pytorch_lightning.loggers import CometLogger
        comet_logger = CometLogger(
                api_key = args.api_key,
                project_name =args. project_name,  
                experiment_name = args.experiment_name,
                save_dir = args.log
        )

        trainer = pl.Trainer(
                num_sanity_val_steps = 0, 
                max_epochs = args.epoch,
                checkpoint_callback = None, 
                logger = comet_logger,
                gpus = args.gpu_ids
            )
 
    else:
        trainer = pl.Trainer(
                num_sanity_val_steps = 0, 
                max_epochs = args.epoch,
                checkpoint_callback = None, 
                gpus = args.gpu_ids
            )
 
    trainer.fit(system)


if __name__ == "__main__":
    args = parseArgs()
    main(args)
