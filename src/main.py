"""
Unified experiment runner for TRACE.

Usage:
  python main.py --params_name criteo_trace
  python main.py --params_name taobao_trace
"""

import argparse
from utils import get_data, set_seed
from metric import metric_mean_and_std
import config
from solver import Solver


def get_solver(params):
    dataset = get_data(params["dataset"])
    pretrain_dataset = dataset["pretrain_dataset"]
    stream = dataset["stream"]
    return Solver(pretrain_dataset, stream, params), dataset


def main():
    parser = argparse.ArgumentParser(description="TRACE experiment runner")
    parser.add_argument(
        "--params_name", type=str, default="criteo_trace",
        help="Experiment configuration name defined in config.py")
    args = parser.parse_args()

    params = config.experiment_params[args.params_name]
    seed_list = [params["seed"]] if isinstance(params["seed"], int) else params["seed"]
    print(f"Experiment: {args.params_name}")
    print(f"Seeds: {seed_list}")

    metric_list = []
    for seed in seed_list:
        set_seed(seed)
        params["current_seed"] = seed
        print(params)

        solver, dataset = get_solver(params)
        print("Pretraining")
        solver.pretrain()
        print("Streaming train and predict")
        results = solver.stream_train_and_predict()
        metric = results["metric"]
        print(params)
        print(metric)
        metric_list.append(metric)

    metric_mean_and_std(metric_list)


if __name__ == "__main__":
    main()
    print("Finished.")