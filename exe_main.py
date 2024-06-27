import pandas as pd
from omegaconf import OmegaConf
from src.arg_process import get_conf
from src.ab_test_pipeline import ab_test_pipline

def main():
    conf = get_conf()
    print(conf)

    data = pd.read_csv(conf.data)
    ab_test_pipline(
        data=data,
        y_col=conf.y_col,
        model=conf.model,
        group_col=conf.group_col,
        num_iterations=conf.train_pram.num_iterations,
        lr=conf.train_pram.lr,
        betas=tuple(OmegaConf.to_container(conf.train_pram.betas)),
        )
    


if __name__ == "__main__":
    main()
