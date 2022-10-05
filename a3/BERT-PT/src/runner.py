import os
import argparse
import json
import glob
import random
import numpy as np
import torch
from collections import namedtuple
import reviewlab
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


class RunManager(object):

    def succ(self, run_config):
        # for test_file in run_config.test_file:
        if not os.path.exists( os.path.join(run_config.output_dir, "predictions.json")):
            return False
        return True
            
    def run_seed(self, config, seed):
        # read in the config file
        run_config = dict(config)
        run_config["seed"] = seed
        run_config["output_dir"] = os.path.join(run_config["output_dir"], str(seed))
        # like C structs
        run_config = namedtuple("run_config", run_config.keys())(*run_config.values())
        # create ft_runs/output_dir
        os.makedirs(run_config.output_dir, exist_ok = True)
        # it the predictions.json is already generated, return
        if self.succ(run_config):
            return
        # write config.json in ft_runs/output_dir
        with open(os.path.join(run_config.output_dir, "config.json"), "w") as fw:
            json.dump(config, fw)

        reviewlab.config.set_seed(run_config)
        # AETrianer
        trainer = getattr(reviewlab, run_config.task.upper() + "Trainer")()
        trainer.train(run_config)
        # test and generate predictions.json
        for test_file in run_config.test_file:
            trainer.test(run_config, test_file)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required = True, type = str)
    parser.add_argument("--seed", required = True, type = int)
    args = parser.parse_args()
    
    mgr = RunManager()
    with open(args.config) as f:
        config = json.load(f)
    mgr.run_seed(config, args.seed)
    