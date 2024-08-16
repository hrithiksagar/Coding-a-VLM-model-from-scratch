# now a way to combine text input with visual input, image tokens should be kept with text tokens. tokenize the text and make it a list, make a place holder for image tokens, use transformers to replace it.abs 

from typing import Dict, List, Tuple, Union, Tuple, Iterable
import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class PaliGemmaProcessor:
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        
        self.image_size = image_size
        self.image_seq_len = num_image_tokens
        
        # tokenizer is described i this link https://github.com/google-research/big_vision/configs/proj/
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # These toekns are used for object detection [Bounding boxes]
        
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range (128)
            ] # These tokens are used for object segmentation
        
        tokenizer.add_tokens(EXTRA_TOKENS)
        
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # we will add the BOS and EOS tokens to the tokenizer
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        
        self.tokenizer = tokenizer

    