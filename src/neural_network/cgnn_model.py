"""
CGNN Model (CLEAN ORIGINAL VERSION)
==================================
Stable loop-based implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGNN(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_dim = config['model']['input_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.output_dim = config['model']['output_dim']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']

        # Feature encoder
        self.feature_encoder = nn.Linear(1, self.hidden_dim)

        # Graph layers
        self.graph_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        # Graph storage
        self.edge_index = None
        self.edge_attr = None

    def forward(self, data):

        x = data.x  # [batch_size, num_features]

        # Save graph once
        if self.edge_index is None:
            self.edge_index = data.edge_index.to(x.device)
            self.edge_attr = data.edge_attr.to(x.device)

        batch_size, num_features = x.shape
        outputs = []

        # 🔹 Process each sample independently (safe & stable)
        for i in range(batch_size):

            sample = x[i]  # [F]

            # Encode features
            h = self.feature_encoder(sample.unsqueeze(1))  # [F, H]

            # Graph propagation
            for layer in self.graph_layers:

                h_new = layer(h)

                # Message passing
                if self.edge_index is not None and self.edge_index.size(1) > 0:

                    messages = torch.zeros_like(h_new)

                    for e in range(self.edge_index.size(1)):
                        src = self.edge_index[0, e].item()
                        dst = self.edge_index[1, e].item()
                        weight = self.edge_attr[e, 0].item()

                        if src < num_features and dst < num_features:
                            messages[dst] += h_new[src] * weight

                    h = h + messages
                else:
                    h = h_new

                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            # Flatten node embeddings
            h_flat = h.reshape(-1)
            outputs.append(h_flat)

        # Stack batch
        h_batch = torch.stack(outputs)

        # Classification
        logits = self.classifier(h_batch)

        return logits, []

    def predict(self, data):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(data)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, data):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(data)
            return F.softmax(logits, dim=1)
