import json
from train import train

if __name__ == "__main__":
    with open(r'train_config.json', encoding="utf-8") as f:
        config = json.load(f)

    train(config)
