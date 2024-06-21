import pandas as pd
from src.arg_process import get_conf
from src.train import train


def main():
    conf = get_conf()
    print(conf)

    data = pd.read_csv(conf.data)
    print(data)

    train()


if __name__ == "__main__":
    main()
