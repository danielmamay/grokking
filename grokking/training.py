from argparse import Namespace
from typing import Sized
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data import get_data_loaders
from model import Transformer


def main(args: Namespace) -> None:
    wandb.init(project="grokking", config=vars(args))
    assert wandb.run is not None
    config = wandb.config
    device = get_device(config.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    train_inputs, train_labels, val_inputs, val_labels, batch_size = load_data(
        config.operation, config.prime, config.training_fraction, config.batch_size, device
    )
    model, optimizer, scheduler, criterion = setup_model(
        config.num_layers, config.dim_model, config.num_heads, config.prime,
        config.learning_rate, config.weight_decay, device,
    )

    n_train = len(train_inputs)
    perm = torch.randperm(n_train, device=device)
    batch_idx = 0

    for step in tqdm(range(config.num_steps)):
        if batch_idx >= n_train:
            perm = torch.randperm(n_train, device=device)
            batch_idx = 0

        idx = perm[batch_idx : batch_idx + batch_size]
        batch_idx += batch_size

        train_step(model, train_inputs[idx], train_labels[idx], optimizer, criterion)
        scheduler.step()

        if step in (1, 10) or step % 100 == 0:
            train_loss, train_acc = evaluate(model, train_inputs, train_labels, criterion)
            val_loss, val_acc = evaluate(model, val_inputs, val_labels, criterion)
            wandb.log(
                {
                    "training/loss": train_loss,
                    "training/accuracy": train_acc,
                    "validation/loss": val_loss,
                    "validation/accuracy": val_acc,
                },
                step=step,
            )


def get_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_data(
    operation: str,
    prime: int,
    training_fraction: float,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    train_loader, val_loader = get_data_loaders(operation, prime, training_fraction, batch_size)
    train_dataset, val_dataset = train_loader.dataset, val_loader.dataset
    assert isinstance(train_dataset, Sized) and isinstance(val_dataset, Sized)
    train_inputs, train_labels = (t.to(device) for t in next(iter(DataLoader(train_dataset, batch_size=len(train_dataset)))))
    val_inputs, val_labels = (t.to(device) for t in next(iter(DataLoader(val_dataset, batch_size=len(val_dataset)))))
    print(f"train_inputs.device: {train_inputs.device}  shape: {train_inputs.shape}")
    actual_batch_size = min(batch_size, len(train_inputs) // 2)
    return train_inputs, train_labels, val_inputs, val_labels, actual_batch_size


def setup_model(
    num_layers: int,
    dim_model: int,
    num_heads: int,
    prime: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[Transformer, Optimizer, torch.optim.lr_scheduler.LRScheduler, torch.nn.CrossEntropyLoss]:
    model = Transformer(
        num_layers=num_layers,
        dim_model=dim_model,
        num_heads=num_heads,
        num_tokens=prime + 2,
        seq_len=4,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=10
    )

    criterion = torch.nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion


def train_step(
    model: Transformer,
    inputs: Tensor,
    labels: Tensor,
    optimizer: Optimizer,
    criterion: torch.nn.CrossEntropyLoss,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)[-1, :, :]
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()


def evaluate(
    model: Transformer,
    val_inputs: Tensor,
    val_labels: Tensor,
    criterion: torch.nn.CrossEntropyLoss,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        output = model(val_inputs)[-1, :, :]
        loss = criterion(output, val_labels).item()
        acc = (torch.argmax(output, dim=1) == val_labels).float().mean().item()
    return loss, acc
