# now a way to combine text input with visual input, image tokens should be kept with text tokens. tokenize the text and make it a list, make a place holder for image tokens, use transformers to replace it.abs 

from typing import Dict, List, Tuple, Union, Tuple, Iterable
import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def process_images(
    images:List[Image.Image], # list of images
    size: Dict[str, int] = None, # size of the image
    resample: Image.Resampling = None, # resampling filter to use when resizing the image
    rescale_factor: float = None, # rescale the image to 0-1
    image_mean: Optional[Union[float, List[float]]] = None, # mean of the image, if None, no normalization is done
    image_std: Optional[Union[float, List[float]]] = None, # standard deviation of the image, if None, no normalization is done
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image = image, size = (height, width), resample = resample) for image in images
    ]
    
    # convert each image into a numpy array
    images = [np.array(image) for image in images]
    
    # rescale the pixel values to 0-1
    images = [rescale(image, scale = rescale_factor) for image in images]
    
    # Normalize the pixel values to have a mean of 0 and standard deviation of 1
    images = [normalize(image, mean = image_mean, std = image_std) for image in images]
    
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    
    return images

def resize(
    height, width = size, # size of the image
    resized_image = image.resize((width, height), resample = resample, reducing_gap=reducing_gap)):    # resample filter to use when resizing the image 
    return resized_image

def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
    image: np.ndarray, 
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],) -> np.ndarray: # Union is used to specify the type of the parameters, in this case, the mean and std can be either a float or an iterable of floats
    
    mean = np.array(mean, dtype=image.dtype) # converting the mean and std to the same data type as the image
    std = np.array(std, dtype=image.dtype) # converting the mean and std to the same data type as the image
    image = (image - mean) / std # normalizing the image by subtracting the mean and dividing by the standard deviation
    return image

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
        Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process) :
        The input text is tokenized normally.
        A ‹bos> token is added at the beginning, and an additional newline token (\n) is appended.
        This newline token is an essential part of the input prompt the model was trained with, so adding it explicitl!
        The tokenized text is also prefixed with a fixed number of ‹image > tokens.
        NOTE: from the paper it looks like the '\n*
        should be tokenized
        separately, but in the HF implementation this is
        ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb 52aad8460elda2f3657
    """
    return f"{image_token * image_seq_len} {bos_token} {prefix_prompt}\n"

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
        
    # method for calling Image and Text data together
    # input is list of texts and images 
    def __call__(
        self, 
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        assert len(text) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts(Texts)" # # assert is used to check if the condition is true, if it is true, the program will continue to run, if it is false, the program will raise an AssertionError
        # work with only 1 image and 1 prompt at a time. 
        # to process these images, we use a special method 
        pixel_values = process_images(
            images, 
            size(self.image_size, self.image_size), # (224, 224) for paligemma the version we are using, https://huggingface.co/google/paligemma-3b-pt-224
            resample = Image.Resampling.BICUBIC, #  resampling filter to use when resizing the image
            rescale_factor = 1/255.0, # rescale the image to 0-1
            # normalizing the image
            image_mean = IMAGENET_STANDARD_MEAN, # mean of the image
            image_std = IMAGENET_STANDARD_STD, # standard deviation of the image
        ) 
        # convert it to a tensor to process for vision model
        # convert the list of numpy arrays to a single numpy array with shape [Batch_size, 3, 224, 224] ~ [Batch_size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis = 0)
        # convert the numpy array to a pytorch tensor
        pixel_values = torch.tensor(pixel_values) 
        
        # prepend a 'self.image_seq_len' number of image tokens to the text/prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_sq_len = self.image_seq_len,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        
        # Returns the input_ids and attention_mask for the model as pytorch tensors
        inputs = self.tokenizer(
            input_strings,
            padding = padding,
            truncation = truncation,
            return_tensors = "pt",
        )
        
        return_data = {"pixel_values ": pixel_values, **inputs} # **inputs is used to unpack the dictionary into the return_data dictionary. 
        return return_data
    
    def process_images(
        images:List[Image.Image], # list of images
        size: Dict[str, int] = None, # size of the image
        resample: Image.Resampling = None, # resampling filter to use when resizing the image
        rescale_factor: float = None, # rescale the image to 0-1
        image_mean: Optional[Union[float, List[float]]] = None, # mean of the image, if None, no normalization is done
        image_std: Optional[Union[float, List[float]]] = None, # standard deviation of the image, if None, no normalization is done
    ) -> List[np.ndarray]:
        height, width = size[0], size[1]
        images = [
            resize(image = image, size = (height, width), resample = resample) for image in images
        ]
        
        # convert each image into a numpy array
        images = [np.array(image) for image in images]
        
        # rescale the pixel values to 0-1
        images = [rescale(image, scale = rescale_factor) for image in images]
        
        # Normalize the pixel values to have a mean of 0 and standard deviation of 1
        images = [normalize(image, mean = image_mean, std = image_std) for image in images]
        
        # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
        images = [image.transpose(2, 0, 1) for image in images]
        
        return images