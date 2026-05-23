import torch
import torch.nn as nn
import torch.nn.functional as F


class IntensityLoss(nn.Module):
    """Pixel-intensity loss (Eq. 8).

    L_int = (1/HW) * ||I_f - max(I_ir, I_vi)||_1

    Encourages the fused image to preserve the brightest pixel intensity
    from either source at every spatial location.
    """

    def forward(
        self, fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor
    ) -> torch.Tensor:
        return F.l1_loss(fused, torch.max(ir, vi))


class TextureLoss(nn.Module):
    """Gradient-texture loss (Eq. 9).

    L_texture = (1/HW) * || |∇I_f| - max(|∇I_ir|, |∇I_vi|) ||_1

    Uses the Sobel operator to measure fine-grained texture information.
    Encourages the fused gradient magnitude to match the richer of the two
    source gradients at every location.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel gradient magnitude, applied per channel."""
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.sobel_x, padding=1)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return mag.view(B, C, H, W)

    def forward(
        self, fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor
    ) -> torch.Tensor:
        return F.l1_loss(self._gradient(fused), torch.max(self._gradient(ir), self._gradient(vi)))


class ContentLoss(nn.Module):
    """Combined content loss (Eq. 7).

    L_content = L_int + alpha * L_texture

    alpha: weight balancing intensity and texture terms (default 10 per paper).
    """

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.intensity = IntensityLoss()
        self.texture = TextureLoss()

    def forward(
        self, fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor
    ) -> torch.Tensor:
        return self.intensity(fused, ir, vi) + self.alpha * self.texture(fused, ir, vi)


class SemanticLoss(nn.Module):
    """Semantic segmentation loss (Eq. 10-12).

    L_semantic = L_main + lambda_aux * L_aux

    Expects logits (not softmax), consistent with F.cross_entropy.

    Parameters
    ----------
    lambda_aux : weight for the auxiliary head loss (default 0.1 per paper).
    ignore_index : label value to ignore in cross-entropy (default 255).
    """

    def __init__(self, lambda_aux: float = 0.1, ignore_index: int = 255):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.ignore_index = ignore_index

    def forward(
        self,
        logits_main: torch.Tensor,
        logits_aux: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loss_main = F.cross_entropy(logits_main, labels, ignore_index=self.ignore_index)
        loss_aux = F.cross_entropy(logits_aux, labels, ignore_index=self.ignore_index)
        return loss_main + self.lambda_aux * loss_aux


class SeAFusionLoss(nn.Module):
    """Full joint loss used to train SeAFusion (Eq. 13).

    L_joint = L_content + beta * L_semantic

    beta grows progressively during training (Eq. 14: beta = gamma * (m - 1)).
    Pass the current beta value at each forward call.

    Parameters
    ----------
    alpha     : texture weight inside ContentLoss (default 10).
    lambda_aux: auxiliary head weight inside SemanticLoss (default 0.1).
    """

    def __init__(self, alpha: float = 10.0, lambda_aux: float = 0.1):
        super().__init__()
        self.content = ContentLoss(alpha=alpha)
        self.semantic = SemanticLoss(lambda_aux=lambda_aux)

    def forward(
        self,
        fused: torch.Tensor,
        ir: torch.Tensor,
        vi: torch.Tensor,
        logits_main: torch.Tensor,
        logits_aux: torch.Tensor,
        labels: torch.Tensor,
        beta: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        l_content = self.content(fused, ir, vi)
        l_semantic = self.semantic(logits_main, logits_aux, labels)
        total = l_content + beta * l_semantic
        return total, {'content': l_content, 'semantic': l_semantic}
