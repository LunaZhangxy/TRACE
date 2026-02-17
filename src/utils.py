from datetime import datetime
import torch
import numpy as np

from data import get_stream
from criteo_data import CriteoData, criteo_dt_ts, SECONDS_A_DAY, SECONDS_AN_HOUR
from taobao_data import TaobaoData, taobao_dt_ts
from model import MLP, CriteoMLP


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


class MovingAverage:

    def __init__(self, default_value=0):
        self.count = 0
        self.value = 0
        self.avg = default_value

    def add(self, value, batch_size=1):
        if value is None:
            return
        self.value += batch_size * value
        self.count += batch_size
        self.avg = self.value / self.count

    def reset(self):
        self.count = 0
        self.value = 0
        self.avg = 0

    def __str__(self):
        return "{:.6f}".format(self.avg)


def get_optimizer(name, params):
    if name == "Adam":
        return lambda model_params: torch.optim.Adam(
            model_params, lr=params["lr"], weight_decay=params["weight_decay"])
    raise ValueError("Unknown optimizer: {}".format(name))


def get_data(dataset_name):
    """Load and prepare dataset for pretraining and streaming evaluation."""
    if dataset_name == "criteo":
        dataset = CriteoData(decision_type="reveal_y",
                             split_ts=10 * SECONDS_A_DAY)
        # Set uniform attribution deadline (d_max = 30 days).
        # Actual conversion signals are encoded in the trajectory d.
        dataset.ts_y = np.ones_like(dataset.ts_y) * SECONDS_A_DAY * 30 + dataset.ts_x
        dataset.reveal(is_test=True)
        for e in criteo_dt_ts:
            dataset.reveal(dataset.ts_x + e)
        pretrain_dataset, stream_dataset = dataset.split()
        stream = get_stream(stream_dataset, dataset.ts_seg)
        return {"pretrain_dataset": pretrain_dataset, "stream": stream}

    elif dataset_name == "taobao":
        dataset = TaobaoData(debug=False)
        dataset.ts_y = np.ones_like(dataset.ts_y) * SECONDS_A_DAY * 3 + dataset.ts_x
        dataset.reveal(is_test=True)
        for e in taobao_dt_ts:
            dataset.reveal(dataset.ts_x + e)
        pretrain_dataset, stream_dataset = dataset.split()
        stream = get_stream(stream_dataset, dataset.ts_seg)
        return {"pretrain_dataset": pretrain_dataset, "stream": stream}

    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")


def set_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


class ProgressInfo:

    def __init__(self, total_step=None, prefix="", log_steps=None):
        self.prefix = prefix
        self.start_dt = datetime.now()
        self.total_step = total_step
        self._step = 0
        self.log_steps = log_steps

    def print_progress(self, msg=""):
        dt = datetime.now() - self.start_dt
        if self.total_step is None:
            print(f"{self.prefix}, step:{self._step}, "
                  f"time:{str(dt).split('.')[0]}, {msg}")
        else:
            step = max(self._step, 1)
            pct = self._step / self.total_step * 100
            eta = dt / step * self.total_step
            print(f"{self.prefix}, step:{self._step}/{self.total_step} "
                  f"[{pct:03.2f}%], time:{str(dt).split('.')[0]}/"
                  f"{str(eta).split('.')[0]}, {msg}")

    def on_log_steps(self):
        if self.log_steps is not None:
            return self._step % self.log_steps == 0
        return False

    def step(self, step_count=1, msg=""):
        self._step += step_count
        if self.on_log_steps():
            self.print_progress(msg)
