import time
import CLIP.clip as clip
from pathlib import Path
from torch.nn import functional as F
import os
import torch
from pathlib import PosixPath


# 加载CLIP模型，这个可以后期更换
dev = 'cuda:0'
clip_model, preprocess = clip.load('ViT-L/14', device=dev)

for subj_idx in range(1, 9):
    subj = "subj0" + str(subj_idx)
    print("subj: ", subj)

    for train_status in range(0, 2):
        if train_status == 0:
            train = True
        else:
            train = False
        tpe = "training" if train else "test"
        path = "/home/mashuxiao/code/43.algonauts23-main_BlobGPT/dataset/algonauts_2023_challenge_data_gpt4_txt_ready" \
               "/{}/{}_split/{}_images".format(subj, tpe, tpe)

        savepath = "/nsddata/algonauts_2023_challenge_data_gpt4_txt_ready_tensor" \
                   "/{}/{}_split/{}_images".format(subj, tpe, tpe)

        for file_path in Path(path).glob('**/*.txt'):
            print("processing ... ", file_path)  # 打印每个 txt 文件的路径

            # 获取文件名（包括扩展名）
            img_name_with_extension = os.path.basename(file_path)
            # 提取文件名（不包括扩展名）
            img_name = os.path.splitext(img_name_with_extension)[0]

            standardPath = PosixPath(savepath)
            standardPath.mkdir(parents=True, exist_ok=True)

            f_save_path = savepath + '/' + img_name
            if PosixPath(f_save_path).exists() and PosixPath(f_save_path).is_file():
                continue

            with open(file_path, 'r') as file:
                content = file.read()

            tokens = clip.tokenize(content).to(dev)
            text_emb = clip_model.encode_text(tokens)
            text_emb = F.normalize(text_emb, dim=-1, p=2)
            text_emb = text_emb.cpu()

            torch.save(text_emb, f_save_path)

            print("processed !")  # 打印每个 txt 文件的路径-

print("END")
