import torch
import torch.nn as nn
from typing import Optional, Tuple

# Config Class 
class SiglipVisionConfig:
    def __init__(
        self, 
        hidden_size = 768,      # size of embedding vector
        intermediate_size=3072, # linear layer size for feedforward network
        num_hidden_layers=12,   # no of layers of vision transformer
        num_attention_heads=12, 
        num_channels=3,         # how chennels of image RGB
        image_size=224,         # Pali gemma comes in only few sizes 
        patch_size=16,          # each image will be divided into patches of patch_size
        ayer_norm_eps = 1e-6, 
        attention_dropout=0.0,  # not using here, its dropout 
        num_image_tokens: int = None, # how many output embedding does this transformer will output?  which is how many image embeddings we have for each image
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
        
        
        
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embeddings = nn.Conv2d( # Conv2d is used to extract patches from the image
            in_channels = config.num_channels, # RGB channels
            out_channels = config.hidden_size, # embedding size, hidden size
            kernel_size = config.patch_size,   # For each output channel we have 1 kernal, so 3. patch size, how big is the patch we need mostly 16
            stride = config.patch_size,         # patch size, image is of going to be 16*16 pixels. we take first group of 3*3 kernal and keep sliding over each channel of input image
            padding = "valid" # i.e., no padding is added to the image
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2 # how many patches we have in the image. number of patches = (224/patch_size) ** 2 patch)_szie = 16/16
        self.num_positions = self.num_patches # + 1 # +1 for the [CLS] token # equal to number of patches we have, to encode info where these patches come from
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim) # position embedding for each patch and [CLS] token, vector of same size of patch, learned embeddings. each of them is added to info extracted from convolution
        self.register_buffer(
            "position_ids", # register buffer is used to store some tensor which is not a parameter of the model
            torch.arange(self.num_positions).expand((1, -1)), # position ids for each patch and [CLS] token  between 0 and num_positions
            persistent=False, # not saved in the state dict
        )
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _,_,height, width = pixel_values.shape # get height and width of the image, [batch_size, channels, height, width], pixel_values came from numpy and numpy loads batch of images with Height and width with tensor of 3 channels, image size is 224*224
        # Convolve the patch_size kernal over the image with no overlapping patches since the 
        # The output of the convolution will have shape [Batch_size, embed_dim, Num_patches_H, Num_patches_W]
        # where Num_patches_H = height // patch_size and Num_patches_W = width // patch_size
        
        patch_embeds = self.patch_embeddings(pixel_values) # extract patches from the image through convolution. tales batch of images anc convert into list of embeddings of size embed_dim
        # [Batch_size, embed_dim, Num_patches_H, Num_patches_W] -> [Batch_size, Num_patches, embed_dim] # total num_patches each of patches with dimensions embed_dim
        # where Num_patches = Num_patches_H * Num_patches_W
        
        embeddings = patch_embeds.flatten(2).transpose(1, 2) # generally the list will be 2*2 matrix, but we want a single array type so we flatten it.
        # flatten the patches and transpose to get [Batch_size, embed_dim, Num_patches] -> [Batch_size, Num_patches, embed_dim] # changing the order
        # Add position embeddings to each patche. 
        # each positional encoding is a vector 
        
        embeddings = embeddings + self.position_embeddings(self.position_ids) # add position embeddings to each patch [0 - 15 patches (total 16 patches)]
        # [batch_size, num_patches, embed_dim]
        
        return embeddings 
        
        
# Vision Transformer Class
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size # hidden size of this embedding vector
        
        self.embeddings = SiglipVisionEmbeddings(config) # Calling Embeddings class, First needs to extract embedding using this class
        self.encoder = SiglipVisionEncoder(config)       # Calling Encoder Class, then run those embedding in this layer i.e., encoder, Because it reminds encoder of transformer
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps) # then layer normalization, #TODO: Add why and Layer norm works (as Umar said he will explain later)
    
    # Forward method is very simple, 
    def forward(self, pixel_values: torch.Tensor): # pixel_values is input image or batch of images
        hidden_states = self.embeddings(pixel_values) # extracting patches from images and loading here and this is done by "SiglipVisionEmbeddings" Class
        last_hidden_state = self.encoder(inputs_embeds=hidden_states) # then take those embedding and run through encoder, which is a list of layers of transformers. includes multi layer attention, layer norm and feed forward network.
        last_hidden_state = self.post_layernorm(last_hidden_state) # then apply layer normalization
        return last_hidden_state
        
        
        
# Vision Model Class        
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config) # Calling Vision Transformer Class
        
    def forward(self, pixel_values):
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)

