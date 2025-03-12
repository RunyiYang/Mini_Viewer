import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, Type
import torchvision
from transformers import AutoProcessor, AutoModel, AutoTokenizer
try:
    import open_clip
except ImportError:
    raise ImportError("open_clip is not installed, install it with `pip install open-clip-torch`")


class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def gui_cb(self, element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id: positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives):]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
    
    def encode_text(self, text):
        """
        Encode text using OpenCLIP.
        This method tokenizes the input text, encodes it with the model,
        and then normalizes the text feature.
        """
        tokenized_text = self.tokenizer(text).to("cuda")
        with torch.no_grad():
            text_feature = self.model.encode_text(tokenized_text)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return text_feature
    
def get_text_feature(network, text):
    # Tokenize and encode the text
    tokenized_text = network.tokenizer(text).to("cuda")
    with torch.no_grad():
        text_feature = network.model.encode_text(tokenized_text)
    
    # Normalize the text feature
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    
    return text_feature


@dataclass
class SigLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: SigLIPNetwork)
    # Use the SigLIP model ID from Hugging Face
    clip_model_pretrained: str = "google/siglip2-base-patch16-512"
    # SigLIP default embedding dimension for the vision encoder is 768.
    clip_n_dims: int = 768
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)



class SigLIPNetwork(nn.Module):
    def __init__(self, config: SigLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            self.config.clip_model_pretrained, torch_dtype=torch.float16,
        ).to("cuda")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.config.clip_model_pretrained)
        self.clip_n_dims = self.config.clip_n_dims
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/siglip2-base-patch16-512")

        self.positives = self.config.positives
        self.negatives = self.config.negatives

    @property
    def name(self) -> str:
        return "siglip_{}".format(self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def encode_image(self, input):
        # Use the processor to preprocess the image.
        inputs = self.processor(images=[input], return_tensors="pt").to("cuda")
        return self.model.get_image_features(**inputs)
    
    def encode_text(self, text):
        """
        Encode text using SigLIP2.
        This method tokenizes the input text, feeds it into the model to obtain text features,
        and then normalizes the features.
        """
        inputs = self.tokenizer(text, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    # Test OpenCLIP text feature extraction
    print("Testing OpenCLIP text feature extraction...")
    openclip_config = OpenCLIPNetworkConfig()
    openclip_net = OpenCLIPNetwork(openclip_config)
    sample_text = "This is a sample text for OpenCLIP."
    openclip_text_feature = openclip_net.encode_text(sample_text)
    print("OpenCLIP text feature shape:", openclip_text_feature.shape)

    # Test SigLIP text feature extraction
    print("\nTesting SigLIP text feature extraction...")
    siglip_config = SigLIPNetworkConfig()
    siglip_net = SigLIPNetwork(siglip_config)
    sample_text_s = "This is a sample text for SigLIP2."
    siglip_text_feature = siglip_net.encode_text(sample_text_s)
    print("SigLIP text feature shape:", siglip_text_feature.shape)