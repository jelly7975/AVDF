# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision
from hrnet.demo.demo_mod import get_person_detection_boxes, box_to_center_scale, get_pose_estimation_prediction

save_frames = 15
input_fps = 30

save_length = s
save_avi = True

failed_videos = []
root = ""

select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
n_processed = 0
for sess in tqdm(sorted(os.listdir(root))):
    for filename in sorted(os.listdir(os.path.join(root, sess))):
        if filename.endswith('.mp4'):
            cap = cv2.VideoCapture(os.path.join(root, sess, filename))
            framen = 0
            while True:
                i, q = cap.read()
                if not i:
                    break
                framen += 1
            cap = cv2.VideoCapture(os.path.join(root, sess, filename))
            framen = int(save_length * input_fps)
            frames_to_select = select_distributed(save_frames, framen)
            save_fps = save_frames // (framen // input_fps)
            numpy_video = []
            success = 0
            frame_ctr = 0
            numpy_heatmap = []
            while True:
                t1_frame = time.time()
                ret, im = cap.read()
                if not ret:
                    break
                if frame_ctr not in frames_to_select:
                    frame_ctr += 1
                    continue
                else:
                    frames_to_select.remove(frame_ctr)
                    frame_ctr += 1

                try:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                except:
                    failed_videos.append((sess, i))
                    break

                device = "cuda:0"
                box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
                box_model.to(device=device)
                box_model.eval()
                input = []
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(im_rgb / 255.).permute(2, 0, 1).float().to(device)
                input.append(img_tensor)
                pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)
                # 姿态估计
                if len(pred_boxes) >= 1:
                    for box in pred_boxes:
                        center, scale = box_to_center_scale(box, 192, 256)
                        landmark, heatmap = get_pose_estimation_prediction(im_rgb, center, scale)
                landmark = np.squeeze(landmark, axis=0)
                heatmap = np.squeeze(heatmap.cpu().numpy(), axis=0)

                numpy_heatmap.append(heatmap)
                t2_frame = time.time()
                during_frame = t2_frame - t1_frame

            np.save(os.path.join(root, sess, filename[:-4] + '_heatmap.npy'), np.array(numpy_heatmap))

            if len(numpy_heatmap) != 15:
                print('Error', sess, filename)

    n_processed += 1
    with open('processed.txt', 'a') as f:
        f.write(sess + '\n')
    print(failed_videos)
