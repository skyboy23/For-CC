'''
把新minigpt4的文本转换为tensor,这个新的minigpt4是来自于LRV的
网址：
Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning
https://github.com/FuxiaoLiu/LRV-Instruction?tab=readme-ov-file
'''


import time
from scripts.CLIP import clip as clip
from pathlib import Path
from torch.nn import functional as F
import os
import torch
from pathlib import PosixPath
import json


# 加载CLIP模型，这个可以后期更换
dev = 'cuda:0'
clip_model, preprocess = clip.load('ViT-L/14', device=dev)

for subj_idx in range(8, 9):
    subj = "subj0" + str(subj_idx)
    print("subj: ", subj)

    for train_status in range(0, 2):
        if train_status == 0:
            train = True
        else:
            train = False
        tpe = "training" if train else "test"
        path = "/home/mashuxiao/code/43.algonauts23-main_BlobGPT/dataset/minigpt4_new_txt" \
               "/{}/{}_split/{}_images".format(subj, tpe, tpe)

        savepath = "/nsddata/minigpt4_new_txt_tensor" \
                   "/{}/{}_split/{}_images".format(subj, tpe, tpe)

        # 读取 JSON 文件
        with open(path + '/caption.json', 'r') as file:
            _data = json.load(file)

            # 遍历条目
            for item in _data:
                image_name = item.get('image')
                image_caption = item.get('answer')

                # if image_name == 'train-9097_nsd-67353':
                #     print("dd")

                print("processing ... ", image_name)

                # # 获取文件名（包括扩展名）
                # img_name_with_extension = os.path.basename(file_path)
                # # 提取文件名（不包括扩展名）
                # img_name = os.path.splitext(img_name_with_extension)[0]

                standardPath = PosixPath(savepath)
                standardPath.mkdir(parents=True, exist_ok=True)

                f_save_path = savepath + '/' + image_name
                if PosixPath(f_save_path).exists() and PosixPath(f_save_path).is_file():
                    continue

                # 假设 context_length 是模型的最大文本长度
                max_length = 230

                # 确保文本不超过最大长度
                if len(image_caption) > max_length:
                    image_caption = image_caption[:max_length]
                # texts = [text[:max_length] for text in image_caption]

                tokens = clip.tokenize(image_caption).to(dev)
                text_emb = clip_model.encode_text(tokens)
                text_emb = F.normalize(text_emb, dim=-1, p=2)
                text_emb = text_emb.cpu()

                torch.save(text_emb, f_save_path)

                print("processed !")  # 打印每个 txt 文件的路径

print("END")
