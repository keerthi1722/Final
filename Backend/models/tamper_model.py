# models/tamper_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Quantum setup ----------------
n_qubits = 4
n_layers = 1

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def quantum_forward(x, weights):
    # Ensure input is float32
    x = x.float()
    weights = weights.float()
    
    outputs = []
    for i in range(x.shape[0]):
        q = qnode(x[i], weights)
        # qnode returns a list of tensors, stack them and ensure float32
        q_stack = torch.stack(q).float()
        outputs.append(q_stack)
    result = torch.stack(outputs)
    # Ensure output is float32
    return result.float()

# ---------------- Hybrid Quantum Model (VGG16-based) ----------------
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Use VGG16 instead of ResNet18
        vgg16 = models.vgg16(weights="IMAGENET1K_V1")
        # Remove the classifier, keep only the features
        self.cnn = vgg16.features
        
        # VGG16 features output is 512x7x7 = 25088 for 224x224 input
        # But after adaptive pooling it becomes 512
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        for p in self.cnn.parameters():
            p.requires_grad = False

        self.dnn = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits),
            nn.Tanh()
        )

        wshape = qml.templates.StronglyEntanglingLayers.shape(
            n_layers=n_layers,
            n_wires=n_qubits
        )
        # Ensure weights are float32
        self.q_weights = nn.Parameter(0.01 * torch.randn(wshape, dtype=torch.float32))

        self.fc = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 512)
        x = self.dnn(x)
        
        # Ensure quantum weights are float32
        q_weights = self.q_weights.float()
        q = quantum_forward(x, q_weights)
        
        # Ensure both are float32 before addition
        x = x.float()
        q = q.float()
        x = x + 0.1 * q
        
        return self.fc(x)
