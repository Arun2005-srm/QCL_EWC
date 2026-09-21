import pytest
import torch

pytest.importorskip("pennylane")
from models.alignment import SpatialAlignment


def test_quantum_backprop_and_state_roundtrip():
    torch.manual_seed(2)
    model = SpatialAlignment(channels=8, qubits=4, grid=2)
    inputs = torch.randn(2, 8, 4, 4)
    output = model(inputs)
    assert output.shape == inputs.shape
    output.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    restored = SpatialAlignment(channels=8, qubits=4, grid=2)
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(output, restored(inputs))
