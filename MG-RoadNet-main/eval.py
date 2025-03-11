import os
import json
import numpy as np
import torch
from thop import profile, clever_format
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from utils.metrics import eval_metrics
from utils.log_save import toString
import utils.log_save as log_save
import logging

from models import MGNet

dataset_Rootdir = 'dataset/'

def eval(config):
    device = torch.device('cuda:0')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = log_save.read_Logconfig()
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    selected = config['train_model']['model'][config['train_model']['select']]

    if selected == 'MG-RoadNet':
        model = MGNet.build_MGRoad(num_classes=config['num_classes'])

    check_point = os.path.join(config['save_model']['save_path'], selected + '_roads.pth')
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
             transforms.Normalize(mean=[0.2304, 0.3295, 0.4405], std=[0.1389, 0.1316, 0.1278])  # gansu
            # transforms.Normalize(mean=(0.4245, 0.4274, 0.3907), std=(0.2887, 0.2822, 0.2938)),
        ]
    )
    model.load_state_dict(torch.load(check_point), False)
    model.cuda()
    model.eval()
    # confuse_matrix
    conf_matrix_test = np.zeros((config['num_classes'], config['num_classes']))

    correct_sum = 0.0
    labeled_sum = 0.0
    inter_sum = 0.0
    unoin_sum = 0.0
    pixelAcc = 0.0
    mIoU = 0.0

    class_precision = np.zeros(config['num_classes'])
    class_recall = np.zeros(config['num_classes'])
    class_f1 = np.zeros(config['num_classes'])
    with open(config['gansu_img_txt'], 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=total_lines):
            image_name, label_name = line.strip().split()
            root_path = dataset_Rootdir
            image_name = os.path.join(root_path, image_name)
            label_name = os.path.join(root_path, label_name)
            label = torch.from_numpy(np.asarray(Image.open(label_name), dtype=np.int32)).long().cuda()

            image = Image.open(image_name)
            image = transform(image).float().cuda()
            # batch_size=1
            image = image.unsqueeze(0)

            output = model(image)
            correct, labeled, inter, unoin, conf_matrix_test = eval_metrics(output, label, config['num_classes'],
                                                                            conf_matrix_test)
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin
            pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
            mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
            for i in range(config['num_classes']):
                # precision of each class
                class_precision[i] = 1.0 * conf_matrix_test[i, i] / conf_matrix_test[:, i].sum()
                # recall of each class
                class_recall[i] = 1.0 * conf_matrix_test[i, i] / conf_matrix_test[i].sum()
                # f1 of each class
                class_f1[i] = (2.0 * class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])

    print('OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(
        pixelAcc, toString(IOU=mIoU), mIoU.mean(), toString(IOU=class_precision), toString(class_recall),
        toString(IOU=class_f1)))

    np.savetxt(os.path.join(config['save_model']['save_path'], selected + '_conf_matrix_test.txt'), conf_matrix_test,
               fmt="%d")

    logger.info('TEST|OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(
        pixelAcc, toString(IOU=mIoU), mIoU.mean(), toString(IOU=class_precision), toString(class_recall),
        toString(IOU=class_f1)))

    input = torch.randn(4, 3, 256, 256).to(device)
    macs, params = profile(model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    logger.info('FLOPs {} | Params {}'.format(macs, params))

    log_save.saveTrainConfig(logfile)
    logger.info('Confusion Matrix\n%s', np.array2string(conf_matrix_test, separator=', ', formatter={'int_kind': lambda x: "%d" % x}))

if __name__ == "__main__":
    with open(r'eval_config.json', encoding='utf-8') as f:
        config = json.load(f)
    eval(config)
