
import torch

def greedy_decode(logits, blank=0):
    preds = torch.argmax(logits, dim=2)  # (T,B)
    preds = preds[:, 0].cpu().numpy()

    decoded = []
    prev = None
    for p in preds:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded
