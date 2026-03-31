
"""
CGNN Data Loader (FINAL OPTIMIZED + ACCURACY FIX)
================================================
Includes:
- Fast loading
- Categorical encoding (CRITICAL)
- Graph fallback (IMPORTANT)
"""

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import pickle

class CGNNDataLoader:

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = None
        self.num_classes = config['model']['output_dim']

    # ========================================
    # LOAD DATA + ENCODE FEATURES
    # ========================================
    def load_data(self):
        print("="*70)
        print("📊 LOADING DATA")
        print("="*70)

        data_path = self.config['data']['causal_variables']
        edges_path = self.config['data']['causal_relationships']

        df = pd.read_csv(data_path)
        edges_df = pd.read_csv(edges_path)

        # 🔥 IMPORTANT: ENCODE CATEGORICAL FEATURES
        print("🔄 Encoding categorical features...")
        
        # Separate target before encoding
        target_col = 'Accident Severity'
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # 🔥 One-hot encode ONLY features
        X = pd.get_dummies(X)
        
        # Recombine
        df = pd.concat([X, y], axis=1)

        print(f"✅ Loaded {len(df)} samples")
        print(f"✅ Loaded {len(edges_df)} edges")

        return df, edges_df

    # ========================================
    # BUILD GRAPH (WITH FALLBACK)
    # ========================================
    def build_graph(self, df, edges_df):
        print("\n🔗 BUILDING CAUSAL GRAPH")
        print("="*70)

        cache_path = Path("data/graph_cache.pt")

        if cache_path.exists():
            print("⚡ Loading cached graph...")
            return torch.load(cache_path)

        feature_names = list(df.columns)
        self.feature_names = feature_names
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        feature_set = set(feature_names)

        edge_list = []
        edge_weights = []

        for _, row in edges_df.iterrows():
            cause = row['cause']
            effect = row['effect']

            if cause not in feature_set or effect not in feature_set:
                continue

            cause_idx = feature_to_idx[cause]
            effect_idx = feature_to_idx[effect]
            strength = row.get('strength', 1.0)

            edge_list.append([cause_idx, effect_idx])
            edge_weights.append(strength)

            if row.get('type') == 'bidirectional':
                edge_list.append([effect_idx, cause_idx])
                edge_weights.append(strength)

        if len(edge_list) == 0:
            print("⚠️ No valid edges → using identity graph")
        
            num_features = len(feature_names)
            edge_list = [[i, i] for i in range(num_features)]
            edge_weights = [1.0] * num_features
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        print(f"✅ Graph built: {edge_index.shape[1]} edges")

        torch.save((edge_index, edge_attr, feature_names), cache_path)

        return edge_index, edge_attr, feature_names

    # ========================================
    # PREPARE DATA
    # ========================================
    def prepare_data(self):
        print("\n🎲 PREPARING DATA")
        print("="*70)

        df, edges_df = self.load_data()

        target_col = 'Accident Severity'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")

        feature_cols = [col for col in df.columns if col != target_col]

        X = df[feature_cols].values
        y = df[target_col].values

        print(f"✅ Features shape: {X.shape}")
        print(f"✅ Target shape: {y.shape}")

        edge_index, edge_attr, feature_names = self.build_graph(
            df[feature_cols], edges_df
        )

        # SPLIT
        train_size = self.config['data']['train_split']
        val_size = self.config['data']['val_split']
        test_size = self.config['data']['test_split']
        random_seed = self.config['data']['random_seed']

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(1-train_size),
            random_state=random_seed,
            stratify=y
        )

        val_ratio = val_size / (val_size + test_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1-val_ratio),
            random_state=random_seed,
            stratify=y_temp
        )

        print("\n✅ Data Split:")
        print(f"Train: {len(y_train)}")
        print(f"Val:   {len(y_val)}")
        print(f"Test:  {len(y_test)}")

        # NORMALIZE
        print("\n🔄 Normalizing...")
        self.scaler.fit(X_train)

        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # CREATE DATA
        train_data = self._create_data(X_train, y_train, edge_index, edge_attr)
        val_data = self._create_data(X_val, y_val, edge_index, edge_attr)
        test_data = self._create_data(X_test, y_test, edge_index, edge_attr)

        print("\n✅ DATA READY\n")

        return train_data, val_data, test_data, feature_names

    def _create_data(self, X, y, edge_index, edge_attr):
        return Data(
            x=torch.tensor(X, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.long),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=X.shape[1]
        )

    def save_scaler(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved")

    def load_scaler(self, path):
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✅ Scaler loaded")
