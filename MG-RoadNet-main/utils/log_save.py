import logging
import os
import time
import json
from datetime import datetime

def initLogger(model_name, dataset_name):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    dataset_name = dataset_name.split('/')[1]

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = "logs"
    if os.path.exists(log_path) is False:
        os.mkdir(log_path)

    log_name = os.path.join(log_path, model_name + '_' + dataset_name + '_' + rq + '.log')
    logfile = log_name

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logfile, logger

def toString(IOU):
    result = '{'
    for i, num in enumerate(IOU):
        result += str(i) + ': ' + '{:.4f}, '.format(num)
    result += '}'
    return result

def write_Logconfig(logpath):
    with open('utils/log_config.json', 'w') as config_file:
        json.dump({'log_path': logpath}, config_file)

def read_Logconfig():
    with open('utils/log_config.json', 'r') as config_file:
        config = json.load(config_file)
        return config['log_path']
def saveTrainConfig(logFile):
    with open('train_config.json', 'r', encoding='utf-8') as file_a:
        data = json.load(file_a)
    selected_data = {
        "num_classes": data["num_classes"],
        "batch_size":data["batch_size"],
        "num_epoch": data["num_epoch"],
        "learning_rate": data["lr"],
        "weight_decay":data["weight_decay"],
        "momentum":data["momentum"],
    }

    with open(logFile, 'a', encoding='utf-8') as file_b:

        file_b.write(f"Training config below:\n")
        for key, value in selected_data.items():
            file_b.write(f" {key}: {value}\n")
        file_b.write('\n')