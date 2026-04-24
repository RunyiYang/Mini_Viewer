"""Lazy feature-query model helpers.

The viewer can work with precomputed query vectors without loading any model.
When model-backed queries are requested, this module lazy-loads only the needed
encoder:

- SigLIP2 / SigLIP for text queries.
- OpenCLIP for text queries.
- DINOv2 for image-query embeddings.

The default SigLIP encoder is ``google/siglip2-so400m-patch16-512``.
The default DINO encoder is ``facebook/dinov2-base``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class CosineClassifier:
    """Cosine scorer for aligned splat features and query embeddings."""

    @staticmethod
    def score(features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if query.ndim == 1:
            query = query[None]
        features = F.normalize(features.float(), dim=-1, eps=1e-6)
        query = F.normalize(query.float(), dim=-1, eps=1e-6)
        scores = features @ query.T
        return scores.max(dim=-1).values


@dataclass
class OpenCLIPNetworkConfig:
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"


@dataclass
class SigLIPNetworkConfig:
    model_name: str = "google/siglip2-so400m-patch16-512"
    max_length: int = 64
    cache_dir: str | Path | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    attn_implementation: str | None = "sdpa"


@dataclass
class DINOv2NetworkConfig:
    model_name: str = "facebook/dinov2-base"
    cache_dir: str | Path | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    attn_implementation: str | None = "sdpa"


def _cache_kwargs(config: Any) -> dict[str, Any]:
    cache_dir = str(config.cache_dir) if getattr(config, "cache_dir", None) is not None else None
    kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": bool(getattr(config, "local_files_only", False)),
        "trust_remote_code": bool(getattr(config, "trust_remote_code", False)),
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) for key, value in batch.items()}


class OpenCLIPNetwork:
    def __init__(self, config: OpenCLIPNetworkConfig | None = None, device: str = "cuda") -> None:
        self.config = config or OpenCLIPNetworkConfig()
        self.device = torch.device(device)
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
            device=self.device,
        )
        self.model = model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.model_name)
        if self.device.type == "cuda":
            self.model = self.model.half()

    @torch.no_grad()
    def encode_text(self, prompts: str | Iterable[str]) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.tokenizer(list(prompts)).to(self.device)
        features = self.model.encode_text(tokens)
        return F.normalize(features.float(), dim=-1, eps=1e-6)


class SigLIPNetwork:
    """SigLIP/SigLIP2 text encoder using Hugging Face Transformers."""

    def __init__(self, config: SigLIPNetworkConfig | None = None, device: str = "cuda") -> None:
        self.config = config or SigLIPNetworkConfig()
        self.device = torch.device(device)

        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        model_id = self.config.model_name
        common_kwargs = _cache_kwargs(self.config)

        print(f"[feature] Loading SigLIP2 text encoder: {model_id}")
        if not self.config.local_files_only:
            print("[feature] Transformers will download/cache the SigLIP2 weights automatically if needed.")

        try:
            self.processor = AutoProcessor.from_pretrained(model_id, **common_kwargs)
            self._processor_accepts_text_keyword = True
        except Exception as exc:
            print(f"[feature] AutoProcessor failed ({exc}); falling back to AutoTokenizer.")
            self.processor = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
            self._processor_accepts_text_keyword = False

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_kwargs = dict(common_kwargs)
        model_kwargs["torch_dtype"] = dtype
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        try:
            self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            try:
                self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
            except TypeError:
                model_kwargs.pop("torch_dtype", None)
                model_kwargs["dtype"] = dtype
                self.model = AutoModel.from_pretrained(model_id, **model_kwargs)

        self.model = self.model.to(self.device).eval()
        print(f"[feature] SigLIP2 ready on {self.device}.")

    @torch.no_grad()
    def encode_text(self, prompts: str | Iterable[str]) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        texts = [str(x) for x in prompts]

        if self._processor_accepts_text_keyword:
            inputs = self.processor(
                text=texts,
                padding="max_length",
                truncation=True,
                max_length=int(self.config.max_length),
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                texts,
                padding="max_length",
                truncation=True,
                max_length=int(self.config.max_length),
                return_tensors="pt",
            )

        inputs = _move_batch_to_device(inputs, self.device)
        if hasattr(self.model, "get_text_features"):
            features = self.model.get_text_features(**inputs)
        else:
            output = self.model.text_model(**inputs)
            features = getattr(output, "pooler_output", output.last_hidden_state[:, 0])
            if hasattr(self.model, "text_projection"):
                features = self.model.text_projection(features)
        return F.normalize(features.float(), dim=-1, eps=1e-6)


class DINOv2Network:
    """DINOv2 image encoder for image-prompt feature queries.

    DINOv2 is visual-only. It does not encode free-form text. Use it with an
    image path in the GUI prompt box, or pass a precomputed ``--query-feature``
    vector.
    """

    def __init__(self, config: DINOv2NetworkConfig | None = None, device: str = "cuda") -> None:
        self.config = config or DINOv2NetworkConfig()
        self.device = torch.device(device)

        from transformers import AutoImageProcessor, AutoModel

        model_id = self.config.model_name
        common_kwargs = _cache_kwargs(self.config)
        print(f"[feature] Loading DINOv2 image encoder: {model_id}")
        if not self.config.local_files_only:
            print("[feature] Transformers will download/cache the DINOv2 weights automatically if needed.")

        self.processor = AutoImageProcessor.from_pretrained(model_id, **common_kwargs)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_kwargs = dict(common_kwargs)
        model_kwargs["torch_dtype"] = dtype
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        try:
            self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            try:
                self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
            except TypeError:
                model_kwargs.pop("torch_dtype", None)
                model_kwargs["dtype"] = dtype
                self.model = AutoModel.from_pretrained(model_id, **model_kwargs)

        self.model = self.model.to(self.device).eval()
        print(f"[feature] DINOv2 ready on {self.device}.")

    @torch.no_grad()
    def encode_image(self, images: str | Path | Iterable[str | Path]) -> torch.Tensor:
        from PIL import Image

        if isinstance(images, (str, Path)):
            image_paths = [images]
        else:
            image_paths = list(images)
        pil_images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = _move_batch_to_device(inputs, self.device)
        output = self.model(**inputs)
        features = getattr(output, "pooler_output", None)
        if features is None:
            features = output.last_hidden_state[:, 0]
        return F.normalize(features.float(), dim=-1, eps=1e-6)


def get_text_feature(network: OpenCLIPNetwork | SigLIPNetwork, text: str) -> torch.Tensor:
    return network.encode_text(text)


def get_image_feature(network: DINOv2Network, image_path: str | Path) -> torch.Tensor:
    return network.encode_image(image_path)
