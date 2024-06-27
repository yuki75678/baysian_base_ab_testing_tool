import argparse
import os
from omegaconf import OmegaConf


def get_args():
    """
    Parse and return command line arguments.

    This function uses argparse to parse command line arguments for the configuration
    file path. It ensures that the configuration file path is provided and exists.

    Returns:
    argparse.Namespace: Parsed command line arguments containing the configuration file path.

    Raises:
    SystemExit: If the configuration file path is not provided or the file does not exist.
    """
    parser = argparse.ArgumentParser(description="Process configuration file path.")

    parser.add_argument("-c", "--conf", help="path of setting file")
    args = parser.parse_args()

    if not os.path.exists(args.conf):
        parser.error(f"The configuration file {args.conf} does not exist.")

    return args


def get_conf():
    """
    Load and return the configuration from the file specified in command line arguments.

    This function uses OmegaConf to load the configuration file provided via command line
    arguments. It handles errors related to file loading.

    Returns:
    omegaconf.dictconfig.DictConfig: Loaded configuration.

    Raises:
    ValueError: If the configuration file fails to load.
    """
    args = get_args()
    try:
        conf = OmegaConf.load(args.conf)
    except Exception as e:
        raise ValueError(f"Failed to load configuration file: {e}")

    return conf
