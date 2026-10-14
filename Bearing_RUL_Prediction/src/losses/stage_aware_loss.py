"""
Stage-aware PHM loss (bản thay thế an toàn).

Đặc điểm:
- Luôn >= 0 (mọi số hạng đều không âm).
- Không dùng max theo batch: nhãn RUL đã được chuẩn hóa về [0, 1] theo từng
  bearing nên stage = 1 - rul được tính trực tiếp, ổn định giữa các batch.
- Phạt nặng hơn khi dự đoán TRỄ (pred > true: báo còn sống lâu hơn thực tế)
  và nặng hơn khi bearing gần hỏng (stage lớn).

Công thức:
    err    = pred - true
    stage  = 1 - clamp(true, 0, 1)
    w      = (1 + alpha * stage) * (1 + beta * [err > 0])
    loss   = mean(w * err^2) + theta * mean(|err|)
"""
import torch
import torch.nn as nn


class StageAwarePHMLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, theta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Đưa về cùng shape [B] để tránh broadcast ngầm [B,1] - [B]
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        err = y_pred - y_true
        stage = 1.0 - torch.clamp(y_true, 0.0, 1.0)
        late = (err > 0).float()

        weight = (1.0 + self.alpha * stage) * (1.0 + self.beta * late)
        loss = (weight * err ** 2).mean() + self.theta * err.abs().mean()
        return loss