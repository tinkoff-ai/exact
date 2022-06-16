import torch
from exact_pytorch import EXACTLoss


X = torch.tensor([-0.25, 0, 0.25]).float()[..., None]
labels = torch.tensor([0, 0, 1]).long()

bias = torch.nn.Parameter(torch.zeros(1))
log_sigma = torch.nn.Parameter(torch.zeros(1))

criterion = EXACTLoss(log_trick=False)
optimizer = torch.optim.SGD([bias, log_sigma], lr=0.01, weight_decay=0)

for step in range(5000):
    binary_logits = X - bias
    logits = torch.cat([torch.zeros_like(binary_logits), binary_logits], -1)
    sigma = log_sigma.exp()
    ratio = logits / sigma
    loss = criterion(ratio, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
    if step % 1000 == 0:
        print(f"Step {step}\tLoss {loss.item():.3f}\tAccuracy {accuracy:.3f}\tBias {bias.item():.3f}\tSigma {sigma.item():.3f}")
print(f"Final accuracy: {accuracy:.3f}")
print(f"Final bias: {bias.item():.3f}")
