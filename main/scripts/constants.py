import torch
import os

CC3M_FOLDER = os.getenv("CC3M_FOLDER", "datasets/CC3M")
DATAFOLDER = os.getenv("DATAFOLDER", "datasets/CC3M")
PREPROCESS_FEATURE_FOLDER = os.getenv("PREPROCESS_FEATURE_FOLDER", None)

WEIGHTFOLDER = os.getenv("WEIGHTFOLDER", "weights")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")
OUTPUT_SUFFIX = os.getenv("OUTPUT_SUFFIX", "")

IS_STAGE2 = os.getenv("IS_STAGE2", False)
IS_STAGE2 = True if IS_STAGE2 in ["True", "true"] else False

IMG_TOKEN_NUM = 8
ALL_IMG_TOKENS = [f"[IMG{i}]" for i in range(IMG_TOKEN_NUM)]
ALL_IMG_TOKENS_STR = "".join(ALL_IMG_TOKENS)

FMRI_TOKEN_NUM = 8
# ALL_FMRI_TOKENS = [f"[FMRI{i}]" for i in range(FMRI_TOKEN_NUM)]
ALL_FMRI_TOKENS = ["<>"]
for i in range(FMRI_TOKEN_NUM - 2):
    ALL_FMRI_TOKENS.append("FM")
ALL_FMRI_TOKENS.append("<>")
ALL_FMRI_TOKENS_STR = "".join(ALL_FMRI_TOKENS)  # '<>FMFMFMFMFMFM<>'

USE_PREFIX_TUNING = False
USE_LORA = False
USE_CFG = True

if IS_STAGE2:
    USE_LORA = True

PRECISION = torch.bfloat16
TRAINABLE_PRECISION = torch.float32

system_prompt = "Give the following images in <Img>ImageContent</Img> format. " \
                    "You will be able to see the images once I provide it to you. " \
                    "Please understanding images."
PROMPT = system_prompt + f"###Human: Describe this image. ###Assistant:"
