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


class TransformerBlock(nn.Module):
    """A single transformer block with multi-head attention and feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)

        return x


class LanguageModel(nn.Module):
    """A simple language model with multiple transformer blocks."""

    def __init__(self, d_model: int, d_ff: int, num_layers: int, num_heads: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads)
            for _ in range(num_layers)
        ])

        # Output layer
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # Embed input tokens
        x = self.embedding(input_ids)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits


def get_model_config(size):
    """Get model configuration based on size."""
    configs = {
        'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
        'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
        'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
        'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
        '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
    }
    return configs[size]


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
