from __future__ import print_function

import argparse
import os
import mmap
import cv2
import time
import numpy as np
from skimage import io
from tqdm import tqdm

tqdm.monitor_interval = 0


def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True
def CreateCrops_Formr(base_dir, crop_type, size, stride):
    os.mkdir(base_dir + '/mr_' + crop_type + '_crops_224')
    os.mkdir(base_dir + '/mr_' + crop_type + '2_label_crops_224')
    crops_file = open(os.path.join(base_dir, '{}_crops_224.txt'.format(crop_type)),
                      'w')

    full_file_path = os.path.join(base_dir, '{}.txt'.format(crop_type))
    full_file = open(full_file_path, 'r')

    def get_num_lines(file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(),
                        0)
        lines = 0
        while buf.readline():
            lines += 1
        buf.close()
        return lines

    failure_images = []
    for name in tqdm(full_file, ncols=100, desc="{}_crops".format(crop_type),
                     total=get_num_lines(full_file_path)):
        name = name.strip("\n")
        image_file = os.path.join(base_dir, '{}'.format(crop_type), name + '.png')
        gt_file = os.path.join(base_dir, '{}'.format(crop_type) + '_labels', name + '.png')

        if not verify_image(image_file):
            failure_images.append(image_file)
            continue

        image = cv2.imread(image_file)
        gt = cv2.imread(gt_file, 0)
        gt = gt.astype(np.uint8) / 255

        if image is None:
            failure_images.append(image_file)
            continue

        if gt is None:
            failure_images.append(image_file)
            continue

        H, W, C = image.shape
        maxx = round((H - size) / stride)
        maxy = round((W - size) / stride)

        name_a = name

        for x in range(maxx + 1):
            for y in range(maxy + 1):
                im_ = image[2+x * stride:2+x * stride + size, 2+y * stride:2+y * stride + size, :]
                gt_ = gt[2+x * stride:2+x * stride + size,2+y * stride:2+y * stride + size]
                crops_file.write('mr' + '/mr_' + crop_type + '_crops_224/{}_{}_{}.png\t'.format(name, x, y))
                crops_file.write('mr' + '/mr_' + crop_type + '2_label_crops_224/{}_{}_{}.png\n'.format(name, x, y))
                cv2.imwrite(base_dir + '/mr_' + crop_type + '_crops_224/{}_{}_{}.png'.format(name, x, y), im_)
                cv2.imwrite(base_dir + '/mr_' + crop_type + '2_label_crops_224/{}_{}_{}.png'.format(name_a, x, y), gt_)
    crops_file.close()
    full_file.close()
    if len(failure_images) > 0:
        print("Unable to process {} images : {}".format(len(failure_images), failure_images))


def main():
    mr_base_dir = '../dataset/mr'

    start = time.perf_counter()


    # For Massachusetts  512
    # Create crops for training
    # CreateCrops_Formr(mr_base_dir, crop_type='train', size=512, stride=494 )
    # # Create crops for validation
    # CreateCrops_Formr(mr_base_dir, crop_type='val', size=512, stride=494)
    # Create crops for test
    # CreateCrops_Formr(mr_base_dir, crop_type='test', size=512, stride=494 )

    # # For Massachusetts  224
    # # Create crops for training
    CreateCrops_Formr(mr_base_dir, crop_type='train', size=224, stride=212 ) # +2
    # Create crops for validation
    CreateCrops_Formr(mr_base_dir, crop_type='val', size=224, stride=212)
    # Create crops for test
    # CreateCrops_Formr(mr_base_dir, crop_type='test', size=224, stride=212 )
    end = time.perf_counter()
    print('Finished Creating crops, time {0}s'.format(end - start))

if __name__ == "__main__":
    main()
