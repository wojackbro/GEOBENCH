from __future__ import annotations

from typing import Dict, Optional

import timm
import torch
import torch.nn as nn


def reduce_backbone_output(out: torch.Tensor | dict | list | tuple) -> torch.Tensor:
    if isinstance(out, dict):
        for key in ["features", "x", "encoder_output", "last_hidden_state"]:
            if key in out:
                out = out[key]
                break
        else:
            out = next(iter(out.values()))
    if isinstance(out, (list, tuple)):
        out = out[-1]

    if out.dim() == 5:
        return out.mean(dim=(2, 3, 4))
    if out.dim() == 4:
        return out.mean(dim=(2, 3))
    if out.dim() == 3:
        return out.mean(dim=1)
    if out.dim() == 2:
        return out
    return out.view(out.size(0), -1)


class TimmEncoder(nn.Module):
    """timm backbone; DINOv2 ViT defaults to img_size=518 — force 224 to match CityLens pipeline."""

    def __init__(self, model_name: str, img_size: int = 224):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        kwargs: dict = {"pretrained": True, "num_classes": 0}
        # timm DINOv2 weights expect 518 unless overridden; we train at 224 like ResNet/CLIP towers.
        if "dinov2" in model_name.lower():
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **kwargs)
        self.feature_dim = getattr(self.backbone, "num_features", None) or getattr(self.backbone, "embed_dim", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        feat = reduce_backbone_output(out)
        if self.feature_dim is None:
            self.feature_dim = feat.size(-1)
        return feat


class PrithviRGBEncoder(nn.Module):
    def __init__(self, backbone_name: str = "prithvi_eo_v2_300", lora_r: int = 8):
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        self.input_adapter = nn.Conv2d(3, 6, kernel_size=1, bias=False)
        self.backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=True,
            bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            num_frames=1,
        )
        self.feature_dim: Optional[int] = None
        self.lora_applied = False

        if lora_r > 0:
            try:
                from peft import LoraConfig, get_peft_model

                cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=max(8, lora_r * 2),
                    lora_dropout=0.05,
                    bias="none",
                    target_modules=["qkv", "q_proj", "k_proj", "v_proj", "proj"],
                )
                self.backbone = get_peft_model(self.backbone, cfg)
                self.lora_applied = True
            except Exception:
                self.lora_applied = False

    def _forward_backbone(self, x: torch.Tensor):
        # Prithvi / terratorch expect 5D (B, C, T, H, W) with C=6 in the *channel* dim (PyTorch NCDHW).
        attempts = [
            x.unsqueeze(2),  # [B, 6, 1, H, W]
            x,
        ]
        last_err: Optional[Exception] = None
        for candidate in attempts:
            try:
                return self.backbone(candidate)
            except Exception as exc:  # pragma: no cover
                last_err = exc
        raise RuntimeError(f"Prithvi forward failed for all input layouts. Last error: {last_err}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(x)
        out = self._forward_backbone(x)
        feat = reduce_backbone_output(out)
        if self.feature_dim is None:
            self.feature_dim = feat.size(-1)
        return feat


class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.LazyLinear(1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, -1e9)
        weights = torch.softmax(logits, dim=-1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class StreetViewRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, pooling: str = "mean", hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.pooling_name = pooling
        self.hidden_dim = hidden_dim
        self.pool: Optional[AttentionPool] = AttentionPool() if pooling == "attention" else None
        self.head = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _pool_views(self, feats: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling_name == "mean":
            if mask is None:
                return feats.mean(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (feats * mask.unsqueeze(-1)).sum(dim=1) / denom
        if self.pool is None:
            raise RuntimeError("Attention pooling requested but attention pool is not initialized")
        return self.pool(feats, mask)

    def forward_features(self, street_views: torch.Tensor, street_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, v, c, h, w = street_views.shape
        feats = self.encoder(street_views.view(b * v, c, h, w))
        feats = feats.view(b, v, -1)
        return self._pool_views(feats, street_mask)

    def forward(self, street_views: torch.Tensor, street_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.forward_features(street_views, street_mask)
        return self.head(feat).squeeze(-1)


class SatelliteRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.head = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.head(feat).squeeze(-1)


class FusionRegressor(nn.Module):
    def __init__(
        self,
        sat_encoder: nn.Module,
        street_encoder: nn.Module,
        pooling: str = "mean",
        fusion_type: str = "late",
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.sat_model = SatelliteRegressor(sat_encoder, hidden_dim=hidden_dim)
        self.street_model = StreetViewRegressor(street_encoder, pooling=pooling, hidden_dim=hidden_dim)
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.gate: Optional[nn.Module] = nn.LazyLinear(2) if fusion_type == "gated" else None
        self.head = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward_features(
        self, image: torch.Tensor, street_views: torch.Tensor, street_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        sat_feat = self.sat_model.forward_features(image)
        stv_feat = self.street_model.forward_features(street_views, street_mask)
        if self.fusion_type == "gated":
            if self.gate is None:
                raise RuntimeError("Gated fusion requested but fusion gate is not initialized")
            fused = torch.cat([sat_feat, stv_feat], dim=-1)
            weights = torch.softmax(self.gate(fused), dim=-1)
            min_dim = min(sat_feat.size(-1), stv_feat.size(-1))
            sat_crop = sat_feat[:, :min_dim]
            stv_crop = stv_feat[:, :min_dim]
            combined = weights[:, :1] * sat_crop + weights[:, 1:] * stv_crop
        else:
            combined = torch.cat([sat_feat, stv_feat], dim=-1)
            weights = None
        return {"satellite": sat_feat, "street": stv_feat, "combined": combined, "weights": weights}

    def forward(self, image: torch.Tensor, street_views: torch.Tensor, street_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.forward_features(image, street_views, street_mask)
        combined = feats["combined"]
        return self.head(combined).squeeze(-1)


def materialize_model(model: nn.Module, branch: str, batch: Dict[str, torch.Tensor], device: torch.device) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if branch == "satellite":
            _ = model(batch["image"].to(device))
        elif branch == "street":
            _ = model(batch["street_views"].to(device), batch["street_mask"].to(device))
        else:
            _ = model(
                batch["image"].to(device),
                batch["street_views"].to(device),
                batch["street_mask"].to(device),
            )
    model.train(was_training)


def make_satellite_encoder(name: str, lora_r: int = 8, img_size: int = 224) -> nn.Module:
    if name == "prithvi_rgb_lora":
        return PrithviRGBEncoder("prithvi_eo_v2_300", lora_r=lora_r)
    if name == "prithvi_rgb_lora_tl":
        return PrithviRGBEncoder("prithvi_eo_v2_300_tl", lora_r=lora_r)
    if name == "resnet50_sat":
        return TimmEncoder("resnet50", img_size=img_size)
    if name == "dinov2_sat":
        return TimmEncoder("vit_base_patch14_dinov2.lvd142m", img_size=img_size)
    raise ValueError(f"Unknown satellite model: {name}")


def make_street_encoder(name: str, img_size: int = 224) -> nn.Module:
    if name == "resnet50":
        return TimmEncoder("resnet50", img_size=img_size)
    if name == "clip_vitb16":
        return TimmEncoder("vit_base_patch16_clip_224.openai", img_size=img_size)
    if name == "dinov2_vitb14":
        return TimmEncoder("vit_base_patch14_dinov2.lvd142m", img_size=img_size)
    if name == "swin_t":
        return TimmEncoder("swin_tiny_patch4_window7_224", img_size=img_size)
    raise ValueError(f"Unknown street model: {name}")
