# -*- coding: utf-8 -*-
import os
import random

root = ''

for action in os.listdir(root):
    print(action)
    for video in os.listdir(os.path.join(root, action)):
        if not video.endswith('.npy'):
            continue
        label = str(video.split('-')[0])
        audio = video.split('_heatmap')[0] + '_cropped.wav'

        annotation_file = 'annotations.txt'
        r = random.randint(0, 9)
        if 0 < r < 9:
            with open(annotation_file, 'a') as f:
                f.write(os.path.join(root, action, video) + ';' + os.path.join(root, action, audio) + ';' + label + ';training' + '\n')
        elif r == 0:
            with open(annotation_file, 'a') as f:
                f.write(os.path.join(root, action, video) + ';' + os.path.join(root, action, audio) + ';' + label + ';validation' + '\n')
        else:
            with open(annotation_file, 'a') as f:
                f.write(os.path.join(root, action, video) + ';' + os.path.join(root, action, audio) + ';' + label + ';testing' + '\n')


