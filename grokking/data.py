from math import ceil
import torch

MODULO_OPERATIONS = {
    "x+y": lambda x, y: x + y,
    "x-y": lambda x, y: x - y,
    "x^2+y^2": lambda x, y: x**2 + y**2,
    "x^2+xy+y^2": lambda x, y: x**2 + x*y + y**2,
    "x^2+xy+y^2+x": lambda x, y: x**2 + x*y + y**2 + x,
    "x^3+xy": lambda x, y: x**3 + x*y,
    "x^3+xy^2+x": lambda x, y: x**3 + x*y**2 + y
}

OPERATIONS = {
    **MODULO_OPERATIONS,
}

def operation_mod_p_data(operation: str, prime: int, eq_token: int, op_token: int):
    """
    x◦y (mod p) for 0 <= x, y < p
    """
    x = torch.arange(prime)
    y = torch.arange(prime)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    if operation in MODULO_OPERATIONS:
        result = OPERATIONS[operation](x, y).remainder(prime)

    inputs = torch.stack([x, op, y, eq], dim=1)
    labels = result

    return inputs, labels

def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
