import argparse
import os
from omegaconf import OmegaConf


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--conf", help="path of setting file")
    args = parser.parse_args()

    if not os.path.exists(args.conf):
        parser.error(f"The configuration file {args.conf} does not exist.")

    return args


def get_conf():
    args = get_args()
    return OmegaConf.load(args.conf)
