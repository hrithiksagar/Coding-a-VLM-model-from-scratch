# Modelling the Text Language Model using GEMMA
# So, this is the 3rd  logic in the series, 1st file is coding the vision aspect
# 2nd part is coding the text aspect i.e., user prompt and image embeddings mixture which returns a final return_data which is a combination of image+text embeddings, used by the final main Language Model Decoder. 
# 3rd file is this Large Langauge Model Decoder. 
# Prompt = given by user and the image

import torch
from torch import nn
from typing import Optional, Tuple, List, Dict, Union
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        