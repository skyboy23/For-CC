import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import gradio as gr
#
# from minigpt4.common.config import Config
# from minigpt4.common.dist_utils import get_rank
# from minigpt4.common.registry import registry
# from minigpt4.conversation.conversation import Chat, CONV_VISION
from pathlib import PosixPath, Path
import time
from sklearn.decomposition import PCA

from timm.data import resolve_model_data_config
import json

# imports modules for registration
# from minigpt4.datasets.builders import *
# from minigpt4.models import *
# from minigpt4.processors import *
# from minigpt4.runners import *
# from minigpt4.tasks import *

# def parse_args():
#     parser = argparse.ArgumentParser(description="Demo")
#     parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
#     parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )
#     args = parser.parse_args()
#     return args

for subj_idx in range(1, 9):
    subj = "subj0" + str(subj_idx)
    print("subj: ", subj)

    # dest_len = 12
    for train_status in range(2):
        if train_status == 0:
            train = True
        else:
            train = False
        tpe = "training" if train else "test"
        path = "/home/mashuxiao/code/43.algonauts23-main_BlobGPT/dataset/algonauts_2023_challenge_data_gpt4/" \
               "{}/{}_split/{}_images/".format(subj, tpe, tpe)
        savepath = "/dev/shm/algonauts_2023_challenge_data_gpt4" \
                   "/{}/{}_split/{}_images/".format(subj, tpe, tpe)

        # 指定文件夹路径
        folder_path = Path(path)  # 替换为你的文件夹路径

        # 遍历文件夹中的所有文件
        for file_path in folder_path.glob('*'):
            if file_path.is_file():
                print(file_path)  # 打印文件路径
                key_name = os.path.splitext(file_path)[0][-2:]
                if key_name == "_6":
                    print("pass")
                    continue

                loaded_tensor = torch.load(file_path).cpu()

                # 将张量重塑为二维形式 [12, 33 * 4096]
                reshaped_tensor = loaded_tensor.view(12, -1)
                reshaped_tensor = reshaped_tensor.permute(1, 0).numpy()
                # 创建 PCA 模型并进行拟合
                pca = PCA(n_components=6)  # 设置主成分数量为 6
                pca.fit(reshaped_tensor)

                # 对数据进行 PCA 变换
                transformed_data = pca.transform(reshaped_tensor)

                # 将压缩后的数据恢复为原始形状 [6, 33, 4096]
                data_tensor = torch.from_numpy(transformed_data).permute(1, 0).reshape(6, 33, 4096).half()

                # 获取文件名（包括扩展名）
                img_name_with_extension = os.path.basename(file_path)
                # 提取文件名（不包括扩展名）
                img_name = os.path.splitext(img_name_with_extension)[0][:-3]

                img_name_dest_len = img_name + "_6"

                savepath = PosixPath(savepath)
                savepath.mkdir(parents=True, exist_ok=True)
                torch.save(data_tensor, savepath / img_name_dest_len)

print("END")
