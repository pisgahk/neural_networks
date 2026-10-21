import torch
from torch import nn


def saada_implementation(
    m: torch.Tensor, f: torch.Tensor, w: torch.Tensor, b: torch.Tensor, lr: float
):
    for i in range(10000):
        pred_fare = m * w + b
        err = pred_fare - f
        loss = (err**2).mean()
        dw = (2 * err * m).mean()
        db = (2 * err).mean()
        w = w - (lr * dw)
        b = b - (lr * db)

        print(
            f"-[{i}]- Weight:{w.item():.2f}, Bias:{b.item():.2f}, Loss:{loss.item():.2f}"
        )
    print("Numpy Implementation...")


def lowest_possible_loss(m: torch.Tensor, f: torch.Tensor):
    m_flat = m.flatten()
    f_flat = f.flatten()

    # Closed-form least-squares solution for:
    # fare = weight * miles + bias
    m_mean = m_flat.mean()
    f_mean = f_flat.mean()

    optimal_w = ((m_flat - m_mean) * (f_flat - f_mean)).sum() / (
        (m_flat - m_mean) ** 2
    ).sum()
    optimal_b = f_mean - optimal_w * m_mean

    optimal_pred = optimal_w * m_flat + optimal_b
    lowest_mse = ((optimal_pred - f_flat) ** 2).mean()

    print("\nLowest possible loss for this linear model:")
    print(f"Optimal weight: {optimal_w.item():.4f}")
    print(f"Optimal bias: {optimal_b.item():.4f}")
    print(f"Lowest possible MSE: {lowest_mse.item():.4f}")


def autograd_implementation(
    m: torch.Tensor,
    f: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    lr: float,
):

    opt = torch.optim.SGD([w, b], lr=lr)
    for step in range(10000):
        prediction = m * w + b
        loss = ((prediction - f) ** 2).mean()
        loss.backward()

        # with torch.no_grad():
        #     w -= lr * w.grad
        #     b -= lr * b.grad
        opt.step()  # w -= lr * w.grad, for every weight.
        opt.zero_grad()  # .grad = None, for every weight.

        w.grad, b.grad = None, None

        print(f"-[{step}]- Weights: {w:.3f}, Biases: {b:.3f} ")
    print("AutoGrad implementation Hogya...")


class FareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, m):
        return self.w * m + self.b


def pytorch_implementation(
    m: torch.Tensor,
    f: torch.Tensor,
    lr: float = 0.0001,
    steps: int = 10000,
    print_every: int = 500,
) -> FareModel:
    """Train and return a linear fare model using PyTorch."""
    m = torch.as_tensor(m, dtype=torch.float32)
    f = torch.as_tensor(f, dtype=torch.float32)
    if m.shape != f.shape:
        raise ValueError("miles and fares must have the same shape")

    model = FareModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(steps):
        loss = loss_fn(model(m), f)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if print_every and (step % print_every == 0 or step == steps - 1):
            print(
                f"-[{step}]- Weight:{model.w.item():.4f}, "
                f"Bias:{model.b.item():.4f}, Loss:{loss.item():.4f}"
            )

    return model


def save_model():
    torch.save(model.state_dict(), "fare_model.pt")


def load_model():
    copy = FareModel()
    copy.load_state_dict(torch.load("fare_model.pt"))
    print(copy.state_dict())


def main():

    miles = torch.tensor(
        [
            [2.50, 9.37, 11.21, 2.48, 3.36, 7.14, 1.38, 0.88, 1.71, 1.00, 1.47, 1.26],
            [0.88, 1.59, 11.40, 1.20, 1.60, 3.44, 1.85, 2.02, 0.60, 1.00, 0.63, 1.33],
            [6.74, 2.88, 0.70, 6.60, 3.74, 2.19, 1.89, 1.95, 0.59, 1.19, 2.13, 4.21],
        ]
    )
    fares = torch.tensor(
        [
            [14.2, 38.0, 43.6, 15.6, 19.8, 45.7, 11.4, 7.9, 10.7, 8.6, 10.7, 9.3],
            [5.8, 9.3, 44.3, 8.6, 12.1, 20.5, 13.5, 21.9, 10.0, 8.6, 6.5, 10.7],
            [29.6, 18.4, 6.5, 33.8, 20.5, 11.4, 11.4, 12.8, 5.8, 9.3, 14.2, 21.2],
        ]
    )

    weights = torch.tensor(1.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    learning_rate = 0.0001

    # saada_implementation(miles, fares, weights, bias, learning_rate)

    # lowest_possible_loss(miles, fares)

    # autograd_implementation(miles, fares, weights, bias, learning_rate)

    pytorch_implementation(miles, fares)


if __name__ == "__main__":
    main()
