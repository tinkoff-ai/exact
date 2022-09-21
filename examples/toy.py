import torch
from exact_pytorch import EXACTLoss, GradientNormalizer


X = torch.tensor([-6, -5, -4, 0, 2]).float()[..., None]
labels = torch.tensor([0, 1, 1, 0, 1]).long()

bias = torch.nn.Parameter(torch.zeros(1))
sigma = 100
criterion = EXACTLoss(disable_batch_norm=True)
optimizer = torch.optim.SGD([bias], lr=0.01, weight_decay=0)
normalizer = GradientNormalizer()

for step in range(5000):
    binary_logits = X - bias
    logits = torch.cat([torch.zeros_like(binary_logits), binary_logits], -1)
    sigma *= 0.99
    loss = criterion(logits, labels, std=sigma)
    loss.backward()
    normalizer([bias])
    optimizer.step()
    optimizer.zero_grad()
    accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
    if step % 1000 == 0:
        print(f"Step {step}\tLoss {loss.item():.3f}\tAccuracy {accuracy:.3f}\tBias {bias.item():.3f}\tSigma {sigma:.3f}")
print(f"Final accuracy: {accuracy:.3f}")
print(f"Final bias: {bias.item():.3f}")
