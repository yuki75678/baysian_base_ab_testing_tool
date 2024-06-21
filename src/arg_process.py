import argparse
from omegaconf import OmegaConf


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--conf", help="path of setting file")
    args = parser.parse_args()

    return args


def get_conf():
    args = get_args()
    return OmegaConf.load(args.conf)
