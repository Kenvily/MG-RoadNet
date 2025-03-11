import os
import cv2
import json
import numpy as np
import torch
import re
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from models import MGNet

dataset_Rootdir = 'dataset/'


def predict(config, logconfig, num_classes):


    device = torch.device('cuda:0')

    selected = config['predict_model']['model'][config['predict_model']['select']]
    if selected == 'TESTNet':
        model = MGNet.build_MGRoad(num_classes=config['num_classes'])

    check_point = os.path.join(config['save_model']['save_path'], selected + '_roads.pth')
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.2304, 0.3295, 0.4405], std=[0.1389, 0.1316, 0.1278]) # gansu
            transforms.Normalize(mean=(0.4245, 0.4274, 0.3907), std=(0.2887, 0.2822, 0.2938)),  # MA
        ]
    )
    model.load_state_dict(torch.load(check_point), False)
    model.cuda()
    model.eval()

    dataset_name = config['gansu_img_txt'].split('/')[1]
    train_time = logconfig['log_path']
    match = re.search(r'\d+', train_time)
    if match:
        train_time = match.group()
    pre_base_path = os.path.join(config['pre_dir'], 'predict_' + dataset_name + '_' + train_time)
    if not os.path.exists(pre_base_path):
        os.makedirs(pre_base_path)
    pre_mask_path = os.path.join(pre_base_path, 'mask')
    if not os.path.exists(pre_mask_path):
        os.makedirs(pre_mask_path)
    pre_vis_path = os.path.join(pre_base_path, 'vis')
    if not os.path.exists(pre_vis_path):
        os.makedirs(pre_vis_path)

    with open(config['gansu_img_txt_224'], 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)

        for line in tqdm(f, total=total_lines):
            image_name, _ = line.strip().split()
            root_path = dataset_Rootdir
            image_name = os.path.join(root_path, image_name)

            image = Image.open(image_name)
            image = transform(image).float().cuda()
            image = image.unsqueeze(0)

            output = model(image)
            _, pred = output.max(1)
            pred = pred.view(config['img_width'], config['img_height'])
            mask_im = pred.cpu().numpy().astype(np.uint8)

            file_name = image_name.split('/')[-1]
            save_label = os.path.join(pre_mask_path, file_name)
            cv2.imwrite(save_label, mask_im)
            save_visual = os.path.join(pre_vis_path, file_name)
            translabeltovisual(save_label, save_visual, num_classes)

def translabeltovisual(save_label, path, num_classes):
    im = cv2.imread(save_label)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            pred_class = im[i][j][0]
            im[i][j] = num_classes[pred_class]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, im)


if __name__ == "__main__":
    with open(r'predict_config.json', encoding='utf-8') as f1:
        config = json.load(f1)
    with open(r'utils/log_config.json', encoding='utf-8') as f2:
        log_config = json.load(f2)
    num = int(config['num_classes'])
    num_classes = [[0, 0, 0], [255, 255, 255]]
    predict(config, log_config, num_classes)
