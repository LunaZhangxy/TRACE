from criteo_data import criteo_dt_ts, SECONDS_AN_HOUR, SECONDS_A_DAY
from taobao_data import taobao_dt_ts


default_config = {
    "device": "cuda",
    "seed": 0,
    "num_workers": 16,
    "weight_decay": 1e-6,
    "log_steps": 500,
    "test_batch_size": 20480,
    "batch_size": 4096,
    "update_steps": 1,
    "pretrain_epochs": 1,
    "lr": 1e-3,
    "optimizer": "Adam",
}


experiment_params = {

    "criteo_ce": {
        "dataset": "criteo",
        "method": "ce",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 2,
    },
    "criteo_oracle": {
        "dataset": "criteo",
        "method": "oracle",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 2,
    },
    "taobao_ce": {
        "dataset": "taobao",
        "method": "ce",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 6,
        "d_nt": len(taobao_dt_ts),
        "nt": taobao_dt_ts,
        "beta": 2,
    },
    "taobao_oracle": {
        "dataset": "taobao",
        "method": "oracle",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 6,
        "d_nt": len(taobao_dt_ts),
        "nt": taobao_dt_ts,
        "beta": 2,
    },
    "criteo_trace": {
        "dataset": "criteo",
        "method": "trace",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_type": "category",
        "d_size": 2,
        "d_nt": 7,
        "nt": criteo_dt_ts,
        "beta": 2,
        "lambda_con": 0.1,
        "completer_ckpt_path": "./pretrain_model/criteo_completer.pt",       
},
    "taobao_trace": {
        "dataset": "taobao",
        "method": "trace",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_type": "category",
        "d_size": 6,
        "d_nt": len(taobao_dt_ts),
        "nt": taobao_dt_ts,
        "beta": 2,
        "lambda_con": 0.1,
        "completer_ckpt_path": "./pretrain_model/taobao_completer.pt",
    },
}

for k in experiment_params:
    for dk in default_config:
        if dk not in experiment_params[k]:
            experiment_params[k][dk] = default_config[dk]
