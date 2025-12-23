# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd


class Video(nn.Module):

    def __init__(self, num_classes=9, im_per_sample=15):
        super(Video, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(17, 64, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True), )
        self.conv1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True), )
        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True), )
        self.conv4 = nn.Sequential(nn.Conv2d(512, 1024, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU(inplace=True), )

        self.conv1d_0 = conv1d_block(1024, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier_1 = nn.Sequential(nn.Linear(128, num_classes),)
        self.im_per_sample = im_per_sample
        
    def forward_features(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.mean([2, 3])
        return x

    def forward_stage(self, x):
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0, 2, 1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    
    def forward_classifier(self, x):
        x = x.mean([-1])
        x1 = self.classifier_1(x)
        return x1
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage(x)
        x = self.forward_classifier(x)
        return x

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True))



class Audio(nn.Module):

    def __init__(self, num_classes=9):
        super(AudioCNNPool, self).__init__()

        input_channels = 10
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)
        
        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )
            
    def forward(self, x):
        x = self.forward_stage(x)
        x = self.forward_classifier(x)
        return x


    def forward_stage(self,x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):   
        x = x.mean([-1])
        x1 = self.classifier_1(x)
        return x1

def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))


class MultiModalCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(MultiModalCNN, self).__init__()

        self.audio_model = Audio(num_classes=num_classes)
        self.visual_model = Video()

        e_dim = 128
        self.fusion = fusion
        self.classifier_1 = nn.Sequential(
                    nn.Linear(e_dim*2, num_classes),
                )


    def forward (self, x_audio, x_visual, name):
        print(name)
        x_audio = self.audio_model.forward_stage(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage(x_visual)

        audio_pooled = x_audio.mean([-1])
        video_pooled = x_visual.mean([-1])

        a = audio_pooled.shape[0]
        device = "cuda:0"
        df_a = pd.read_csv('.csv')
        name_to_value_a = df_a.set_index('audio_name')['audio_norm'].to_dict()
        audio_value = torch.zeros((a, 1), device=device)
        df_v = pd.read_csv('.csv')
        name_to_value_v = df_v.set_index('video_name')['video_norm'].to_dict()
        video_value = torch.zeros((a, 1), device=device)
        for i, single_name in enumerate(name):
            if single_name in name_to_value_a:
                audio_norm = name_to_value_a[single_name]
            else:
                print(f"Name {single_name} not found in the CSV file.")
            if single_name in name_to_value_v:
                video_norm = name_to_value_v[single_name]
            else:
                print(f"Name {single_name} not found in the CSV file.")

            audio_weight = torch.tensor((2 * audio_norm) / (audio_norm + video_norm), device=device)
            video_weight = torch.tensor((2 * video_norm) / (audio_norm + video_norm), device=device)
            audio_value[i] += audio_weight
            video_value[i] += video_weight

        audio_feature = audio_value * audio_pooled
        video_feature = video_value * video_pooled
        x = torch.cat((audio_feature, video_feature), dim=-1)
        x1 = self.classifier_1(x)
        return x1
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
