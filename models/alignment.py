import math

import torch
from torch import nn
import torch.nn.functional as F


class SpatialAlignment(nn.Module):
    """A shared coarse spatial correction; no task-specific parameters."""

    def __init__(self, channels=256, qubits=4, depth=2, grid=4, kind="quantum", gated=False):
        super().__init__()
        if qubits < 2 or depth < 1 or grid < 1 or kind not in {"quantum", "classical"}:
            raise ValueError("Require qubits>=2, depth>=1, grid>=1 and quantum/classical kind")
        self.grid, self.kind = grid, kind
        self.gate = nn.Parameter(torch.tensor(-2.0)) if gated else None
        self.down = nn.Conv2d(channels, qubits, 1)
        self.up = nn.Conv2d(qubits, channels, 1)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)
        if kind == "quantum":
            import pennylane as qml
            self.angles = nn.Parameter(torch.randn(depth, qubits, 3) * .1)
            device = qml.device("default.qubit", wires=qubits, shots=None)

            @qml.qnode(device, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                for wire in range(qubits):
                    qml.RY(inputs[..., wire], wires=wire)
                for layer in range(depth):
                    for wire in range(qubits):
                        qml.Rot(*weights[layer, wire], wires=wire)
                    for wire in range(qubits - 1):
                        qml.CNOT(wires=[wire, wire + 1])
                    if qubits > 2:
                        qml.CNOT(wires=[qubits - 1, 0])
                return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(qubits))

            self.circuit = circuit
        else:
            self.bottleneck = nn.Sequential(nn.Linear(qubits, qubits), nn.Tanh(), nn.Linear(qubits, qubits), nn.Tanh())

    def forward(self, features):
        pooled = F.adaptive_avg_pool2d(features, self.grid)
        angles = math.pi * self.down(pooled).tanh()
        batch, channels, _, _ = angles.shape
        rows = angles.permute(0, 2, 3, 1).reshape(-1, channels)
        if self.kind == "quantum":
            # Small circuits run on CPU even when SAM uses CUDA. Copies preserve
            # autograd, so gradients return to the CUDA projection/parameters.
            output = self.circuit(rows.to("cpu"), self.angles.to("cpu"))
            rows = torch.stack(output, dim=-1).to(device=features.device, dtype=features.dtype)
        else:
            rows = self.bottleneck(rows)
        correction = self.up(rows.reshape(batch, self.grid, self.grid, channels).permute(0, 3, 1, 2))
        correction = F.interpolate(correction, features.shape[-2:], mode="bilinear", align_corners=False)
        return features + (self.gate.sigmoid() if self.gate is not None else 1) * correction
