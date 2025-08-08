import torch
import torch.nn as nn
from typing import Optional, Tuple

# -------1--------
# 1. Importing Libraries and defining the config class
# This is the main file for the vision model, which is a transformer based model that takes images as input and outputs a list of embeddings for each image. 
# The model is made up of multiple layers of transformer model, each layer has multi head attention, layer normalization and feed forward network. Each layer has residual connection, which is a skip connection that adds the input of the layer to the output of the layer. This is done to avoid vanishing gradient problem and to make the model more robust.
# Config Class 
class SiglipVisionConfig:
    def __init__(
        self, 
        hidden_size = 768,      # size of embedding vector
        intermediate_size=3072, # linear layer size for feedforward network
        num_hidden_layers=12,   # no of layers of vision transformer
        num_attention_heads=12, # number of attention heads in the multi head attention
        num_channels=3,         # how chennels of image RGB
        image_size=224,         # Pali gemma comes in only few sizes, 224, 448, 896.... any image will firts gets resized to
        patch_size=16,          # each image will be divided into patches of patch_size
        layer_norm_eps = 1e-6, 
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

# -----2--------
# 2 Vision Model Class        
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config) # Calling Vision Transformer Class, vision_model is made up of transformer  
        
    def forward(self, pixel_values): # takle pixel values of image via numpy --> converted to array. 
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) # Vision_model will take this image and convert it to Batch size, Num_patches, Embed_Dim, which is a list of patches with embedding vector of size embed_dim. Embed_dim == each vector of fixed dimention. 
    # Basically, vision_model will take list of images of BATCH SIZE, convert them into list of embeddings of size Embed_dim, which is a vector of fixed size, and then return it.

        
# -------3--------        
# 3 Vision Transformer Class
class SiglipVisionTransformer(nn.Module): # this is the a torcvh layer where we pass the config of size of embedding vector and then run the model
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size # hidden size of this embedding vector
        
        self.embeddings = SiglipVisionEmbeddings(config) # Calling Embeddings class, First needs to extract embedding using this class
        self.encoder = SiglipEncoder(config)       # Calling Encoder Class, then run those embedding in this layer i.e., encoder, Because it reminds encoder of transformer. this is a series of layers of transformer. Main part of the transformer encoder model, which is a series of layers of transformer. Each layer has multi head attention, layer normalization and feed forward network.
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps) # then layer normalization, reason is that, added a batch norm notes, check it. 
    
    # Forward method is very simple, we take this image, which is a a batch of images and convert them into batch of embeddings, basiclaly extrtacting patches of images. Take these imagesm, run a convolution, flatten them to get patches then flatten them and add positional embeddings to each patch, then run those patches through encoder, which is a series of layers of transformer.
    def forward(self, pixel_values: torch.Tensor): # pixel_values is input image or batch of images
        hidden_states = self.embeddings(pixel_values) # extracting patches from images and loading here and this is done by "SiglipVisionEmbeddings" Class
        last_hidden_state = self.encoder(inputs_embeds=hidden_states) # then take those embedding and run through encoder, which is a list of layers of transformers. includes multi layer attention, layer norm and feed forward network.
        last_hidden_state = self.post_layernorm(last_hidden_state) # then apply layer normalization
        return last_hidden_state 


# --------4-----------
# 4. this is the embedding layer. 
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size # how big is the image
        self.patch_size = config.patch_size # how big is the batCH THAT WE WANT TO GET FROM THe IMAGE, here we are getting 16, so 16*16 puixels. 
        
        self.patch_embeddings = nn.Conv2d( # Conv2d is used to extract patches from the image
            in_channels = config.num_channels, # RGB channels, 3
            out_channels = config.hidden_size, # embedding size, hidden size
            kernel_size = config.patch_size,   # For each output channel we have 1 kernal, so 3. patch size, how big is the patch we need mostly 16. Kernal is something that is used to extract features from the image, it is a small matrix that is convolved over the image to extract features. More like convolutional Kernal, say 3*3, which will pass through the image and extract output eatures from it. For each output channel there is one channel, here in this case = 3 each for R , G and B. 
            stride = config.patch_size,        # to avoid ovberlap of kernal, when kernal is moving, how much distc does this should move? is what said by stride like in CNN. patch size, image is of going to be 16*16 pixels. we take first group of 3*3 kernal and keep sliding over each channel of input image
            padding = "valid" # i.e., no padding is added to the image
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2 # how many patches we have in the image. number of patches = (224/patch_size) ** 2 patch)_szie = 16/16
        self.num_positions = self.num_patches # + 1 # +1 for the [CLS] token # equal to number of patches we have, to encode info where these patches come from
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim) # position embedding for each patch and [CLS] token, vector of same size of patch, learned embeddings. each of them is added to info extracted from convolution
        self.register_buffer(
            "position_ids", # register buffer is used to store some tensor which is not a parameter of the model
            torch.arange(self.num_positions).expand((1, -1)), # position ids for each patch and [CLS] token  between 0 and num_positions -1
            persistent=False, # not saved in the state dict
        )
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _,_,height, width = pixel_values.shape # pixel values came from numpy, Numpy l;oads a batch of images with Batch ssize with its channels, height and weight. get height and width of the image, [batch_size, channels, height, width], pixel_values came from numpy and numpy loads batch of images with Height and width with tensor of 3 channels, image size is 224*224
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

# ----5---------
# 5. Encoder Class  
"""
Made up of multiple latyers of transformer model, asrchitecture is like this:
Input -> Layer Norm -> Multi Head Attention -> Residual Connection -> Layer Norm -> Feed Forward Network -> Residual Connection -> Output
SiglipVisionEmbeddings gives image patches using a convolutional layer, gets vector embedding which is added to vector called posiitonal embeddings which is learned embeddings, then this stuff is fed to "encoder". 

in short, SiglipVisionEmbeddings extracts patches from the image and converts them into a list of embeddings, which is then passed to the encoder. The encoder is made up of multiple layers of transformer model, each layer has multi head attention, layer normalization and feed forward network. Each layer has residual connection, which is a skip connection that adds the input of the layer to the output of the layer. This is done to avoid vanishing gradient problem and to make the model more robust.
"""
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config) # Calling Attention Class # 
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) # Layer normalization
        self.mlp = SiglipMLP(config) # <6> Calling MLP Class # this is multi layer perceptron layer 
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) # Layer normalization layer
    
    # main encoder logic forward pass. 
    def forward( self, hidden_states: torch.Tensor) -> torch.Tensor: # hidden states from previous layer
        # residual: [Batch_size, Num_patches, Embed_dim] connection, send the input we feed to the encoder. 
        residual = hidden_states # Skip connection
        
        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim] then we run the input via layer normalization, the shape doesnt get changed. 
        hidden_states = self.layer_norm1(hidden_states) # Layer normalization; its like the gaussian wiht mean 0 and variance 1 
        
        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states) # Self Attention, this is delat later, doesnt now chnage the input embeddings size, just gives ocntextualized embeddings. Consider this as blackbox as of now. 
        
        #[Batch_size, Num_patches, Embed_dim], this is a skip opr residual connections, basiclaly an ADDITION of hidden state and self attention./
        hidden_states = residual + hidden_states # Residual connection, Skip connection + output of self attention
        
        #residual: [Batch_size, Num_patches, Embed_dim]
        residual = hidden_states
        
        hidden_states = self.layer_norm2(hidden_states) # Layer normalization, another linear layer normalization that doesnt chnage the shape. 
        
        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = self.mlp(hidden_states) # Feed Forward Network, take each input embedding and transform it to another embedding, here there is not mixing between them, each of them is transformed independently, has more degrees of freedom to learn, allows to prepare the sequence of patches for next layers and added non linearity which allows to learn complex patterns. 
        # Not very easy to explain, series of linea rlayers that takes each input embeddings and transforms it independently each other. There i no mixing between the patches, each patch is transformed independently, so models have more degree of freedom to learn on multiple tasks. 
        
        # [Batch_size, Num_patches, Embed_dim]
        hidden_states = residual + hidden_states # Residual connection, "Skip connection + output of feed forward network"
        
        return hidden_states
    
#-----6--------
""" this is the just like vanilla transformer, made up of 2 layers. 
1st layer: takes up tokens or patches of images or each embeddings: and expands them inermediate size, chosen as 3 or 4 times the hidden zie
2nd layer: then apply a non linearlrity
3rd step: compress it to hidden size dimension again
but why to do this?
Because it allows the model to learn more complex patterns and relationships between the patches, by expanding the embeddings to a larger size, applying a non-linearity, and then compressing it back to the hidden size, the model can learn more complex relationships between the patches and the image as a whole.
"""
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # fc = fully connected layer
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size) # Linear layer, input size is hidden size and output size is intermediate size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size) # Linear layer, input size is intermediate size and output size is hidden size
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Intermediate_size]
        hidden_states = self.fc1(hidden_states)
        
        # hidden_states: [Batch_Size, Num_patches, Intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh") # GELU activation function, non linear activation function which is used in transformers and is better than ReLU and Sigmoid functions because it is smooth, non linear and has less vanishing gradient problem than ReLU. It is used to add non linearity to the model. in GELU anything that is negative becomes 0 and anything that is positive remains same.
        
        #[Batch_size, Num_patches, Intermediate_size] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states
        


        
#----7------
# Attention Mechanism
class SiglipAttention(nn.Module):
    """ Multi head attention from 'Attention is All You Need' paper """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # scale factor, Equivalent to 1/Sqrt(self.head_dim). Why -0.5? Because it is used in the paper
        self.dropout = config.attention_dropout # never saw it used in Pali Gemma, but otehr models uses it , but can think its non existsnet for now
        
        # Query, Key, Value Projection weights 
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim) 
    
    # output of layer normalizaton is fed to forward pass of attention mechanism as input    
    def forward(self, hidden_states: torch.Tensor,) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # hidden_states: [Batch_size, Num_patches, Embed_dim], it means we have a batch of images, each image is made up of num_patches, each oatch is represented by vector of size embed dimention 1024*1024, num patches can also be thought as sequence length. 
        # First thing is take input and ran it through wq, wk and wv thier outputs are query key and vector. 
        batch_size, seq_len, _ = hidden_states.size() # get batch size, sequence length and embedding size of the hidden states 
        
        # query_states: [Batch_size, Num_patches, Embed_dim]
        query_states = self.q_proj(hidden_states) # project the hidden states to query space
        
        # key_states: [Batch_size, Num_patches, Embed_dim]
        key_states = self.k_proj(hidden_states) # project the hidden states to key space
        
        # value_states: [Batch_size, Num_patches, Embed_dim]    
        value_states = self.v_proj(hidden_states) # project the hidden states to value space
        
        # splitting embed_dim into smaller dimenstions, (num_heads * head_dim = embed_dim). Just grouping differently 
        # query_states: [Batch_size, Num_patches, Num_heads, Head_dim]
        # splitting embed dimensions into smlaler parts called head ismentions, and transpose it, do it for QKV 
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape the query states to [Batch_size, Num_patches, Num_heads, Head_dim] and then transpose to [Batch_size, Num_heads, Num_patches, Head_dim]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape the key states to [Batch_size, Num_patches, Num_heads, Head_dim] and then transpose to [Batch_size, Num_heads, Num_patches, Head_dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape the value states to [Batch_size, Num_patches, Num_heads, Head_dim] and then transpose to [Batch_size, Num_heads, Num_patches, Head_dim]
        
        # calculate attention weights, using formula Q.K^T / sqrt(d_k). attn_weights: [Batch_size, Num_heads, Num_patches, Num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale) # matrix multiplication of query and key states and then scale it by self.scale
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should have the shape {(Batch_size, Num_heads, Num_patches, Num_patches)}, but got {attn_weights.size()}"
            )
        
        # Apply the softmax row-wise to calculate the attention weights
           # attn_weights: [Batch_size, Num_heads, Num_patches, Num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) # apply softmax to the last dimension of the attention weights
        
        # Apply dropout to the attention weights only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # apply dropout to the attention weights only during training to reduce overfitting
        
        # Multiply the attention weights with the value states to get the context matrix
        # context: [Batch_size, Num_heads, Num_patches, Head_dim]
        attn_output = torch.matmul(attn_weights, value_states) # matrix multiplication of attention weights and value states
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should have the shape {(Batch_size, Num_heads, Num_patches, Head_dim)}, but got {attn_output.size()}"
            )
        
        # we will transpose back
        # [batch-size, Num_heads, Num_patches, Head_dim] -> [batch-size, Num_patches, Num_heads, Head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous() # transpose the attention output to [batch-size, Num_patches, Num_heads, Head_dim] and then reshape it to [batch-size, Num_patches, Embed_dim], contiguous() is used to make sure that the tensor is stored in a contiguous chunk of memory without computatin overhead
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim) # reshape the attention output to [batch-size, Num_patches, Embed_dim]
        
        #[Batch_Size,Num_Patches,Embed_dim]
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
        
#-----8----
""" 8. Class for making a sequence of Enocoder layer for making input for the model, this output will be input for the model in next layer..... 
Basiclaly this is the sequence of EncoderLayersm, i.e., in the fucntion "SiglipEncoderLayer", we wrote the main logic for encoder layer, tbus this layer should be repeated multiple times to make a sequence of encoder layers, which is done by this class. So the output of one will be input for the next layer, and so on.
"""
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)] # the number of times it should repeate is based on the size of the hidden layers. 
        )
        
    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor: # sinmple, says we have input, pass it to the first layer, then its ouptut of this layer will become input to the next layer. 
        # input_embeds: [Batch_size, Num_patches, Embed_Dim]
        
        hidden_states = input_embeds
        
        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -› [Batch_Size, Num_Patches, Embed_Dim] # no chnage in the sdhape of the input, just the values are changed
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
            
# now a way to combine text input with visual input, image tokens should be kept with text tokens. tokenize the text and make it a list, make a place holder for image tokens, use transformers to replace it.abs 