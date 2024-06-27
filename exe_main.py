import pandas as pd
from omegaconf import OmegaConf
from src.arg_process import get_conf
from src.ab_test_pipeline import ab_test_pipeline


def load_configuration():
    """
    Load and return the configuration from the provided configuration file.

    Returns:
    omegaconf.dictconfig.DictConfig: Loaded configuration.

    Raises:
    ValueError: If the configuration file fails to load.
    """
    try:
        conf = get_conf()
        return conf
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def load_data(file_path):
    """
    Load and return the data from the provided CSV file path.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data frame.

    Raises:
    ValueError: If the data file fails to load.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Error reading data from {file_path}: {e}")


def run_ab_test(conf, data):
    """
    Run the A/B testing pipeline with the provided configuration and data.

    Parameters:
    conf (omegaconf.dictconfig.DictConfig): The configuration object.
    data (pd.DataFrame): The data frame to use for A/B testing.

    Raises:
    ValueError: If there is an error running the A/B testing pipeline.
    """
    try:
        ab_test_pipeline(
            data=data,
            y_col=conf.y_col,
            model=conf.model,
            group_col=conf.group_col,
            num_iterations=conf.train_pram.num_iterations,
            lr=conf.train_pram.lr,
            betas=tuple(OmegaConf.to_container(conf.train_pram.betas)),
        )
    except Exception as e:
        raise ValueError(f"Error running A/B testing pipeline: {e}")


def main():
    """
    Main function to run the A/B testing pipeline.

    This function loads the configuration, reads the data, and runs the A/B testing pipeline
    using the specified parameters from the configuration.
    """
    try:
        conf = load_configuration()
        data = load_data(file_path=conf.data)
        run_ab_test(conf=conf, data=data)
    except Exception as e:
        print(f"Critical error in main execution: {e}")


if __name__ == "__main__":
    main()
