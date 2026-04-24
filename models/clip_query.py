"""Lazy language-query helpers.

Nothing here imports heavyweight CLIP/SigLIP dependencies at module import time, so
CPU-only viewer usage remains lightweight.

The default SigLIP encoder is SigLIP2 So400m patch16 512:
``google/siglip2-so400m-patch16-512``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


class CosineClassifier:
    """Cosine scorer for language features and query embeddings."""

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
    # User-requested default: SigLIP2-so400m-p16-512.
    # Hugging Face checkpoint id uses "patch16" instead of "p16".
    model_name: str = "google/siglip2-so400m-patch16-512"
    max_length: int = 64
    cache_dir: str | Path | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    attn_implementation: str | None = "sdpa"


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
    """SigLIP/SigLIP2 text encoder using Hugging Face Transformers.

    ``from_pretrained`` downloads the model/processor automatically on first use
    and then reuses the local Hugging Face cache on later runs.
    """

    def __init__(self, config: SigLIPNetworkConfig | None = None, device: str = "cuda") -> None:
        self.config = config or SigLIPNetworkConfig()
        self.device = torch.device(device)

        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        model_id = self.config.model_name
        cache_dir = str(self.config.cache_dir) if self.config.cache_dir is not None else None
        common_kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": bool(self.config.local_files_only),
            "trust_remote_code": bool(self.config.trust_remote_code),
        }
        common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}

        print(f"[language] Loading SigLIP2 text encoder: {model_id}")
        if not self.config.local_files_only:
            print("[language] Transformers will download/cache the SigLIP2 weights automatically if needed.")

        try:
            self.processor = AutoProcessor.from_pretrained(model_id, **common_kwargs)
            self._processor_accepts_text_keyword = True
        except Exception as exc:
            print(f"[language] AutoProcessor failed ({exc}); falling back to AutoTokenizer.")
            self.processor = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
            self._processor_accepts_text_keyword = False

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_kwargs = dict(common_kwargs)
        # ``torch_dtype`` works on Transformers 4.x. If a future version only
        # accepts ``dtype``, the fallback below handles that.
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
        print(f"[language] SigLIP2 ready on {self.device}.")

    @torch.no_grad()
    def encode_text(self, prompts: str | Iterable[str]) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        texts = [str(x) for x in prompts]

        # SigLIP2 training used lowercased text and max_length=64. The official
        # processor handles the normal preprocessing; the explicit arguments make
        # behavior stable across Transformers versions.
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

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        if hasattr(self.model, "get_text_features"):
            features = self.model.get_text_features(**inputs)
        else:
            output = self.model.text_model(**inputs)
            features = getattr(output, "pooler_output", output.last_hidden_state[:, 0])
            if hasattr(self.model, "text_projection"):
                features = self.model.text_projection(features)
        return F.normalize(features.float(), dim=-1, eps=1e-6)


def get_text_feature(network: OpenCLIPNetwork | SigLIPNetwork, text: str) -> torch.Tensor:
    return network.encode_text(text)
