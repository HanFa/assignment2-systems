import torch
import torch.nn as nn


class ToyModel(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ToyModel(20, 5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Create dummy data
    batch_size = 32
    inputs = torch.randn(batch_size, 20).to(device)
    targets = torch.randint(0, 5, (batch_size,)).to(device)

    # Dictionary to store intermediate outputs
    intermediate_outputs = {}


    # Register forward hooks to capture intermediate outputs
    def make_hook(name):
        def hook(module, input, output):
            intermediate_outputs[name] = output

        return hook


    model.fc1.register_forward_hook(make_hook('fc1'))
    model.ln.register_forward_hook(make_hook('ln'))

    print("=" * 60)
    print("Training with torch.autocast")
    print("=" * 60)

    # Training loop with autocast
    for step in range(3):
        print(f"\nStep {step + 1}:")
        print("-" * 60)

        optimizer.zero_grad()

        # Use autocast for mixed precision
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # Print model parameter dtypes within autocast context
            print("\n1. Model parameter dtypes (within autocast context):")
            for name, param in model.named_parameters():
                print(f"   {name}: {param.dtype}")

            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, targets)

            # Print intermediate output dtypes
            print("\n2. Output of first feed-forward layer (fc1):")
            print(f"   dtype: {intermediate_outputs['fc1'].dtype}")

            print("\n3. Output of layer norm (ln):")
            print(f"   dtype: {intermediate_outputs['ln'].dtype}")

            print("\n4. Model's predicted logits:")
            print(f"   dtype: {logits.dtype}")

            print("\n5. Loss:")
            print(f"   dtype: {loss.dtype}")
            print(f"   value: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Print gradient dtypes
        print("\n6. Model's gradients:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"   {name}.grad: {param.grad.dtype}")

        optimizer.step()

        print()

    print("=" * 60)
    print("Training completed")
    print("=" * 60)
