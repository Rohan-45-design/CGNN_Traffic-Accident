import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .logger import TrainingLogger
from .checkpoint_manager import CheckpointManager


class CGNNTrainer:

    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # ❌ DO NOT define loss here (no train_data yet)
        self.criterion = None

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Batch size
        self.batch_size = config['training'].get('batch_size', 32)

        # Logger
        self.logger = TrainingLogger(config['output']['logs_dir'])

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config['training']['checkpoint_dir'],
            save_best_only=True
        )

    def _create_optimizer(self):
        optimizer_name = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        if optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['training']['scheduler_factor'],
            patience=self.config['training']['scheduler_patience']
        )

    # ========================================
    # TRAIN (MINI-BATCH)
    # ========================================
    def train_epoch(self, train_data):

        self.model.train()

        X = train_data.x
        y = train_data.y

        num_samples = X.size(0)

        total_loss = 0
        correct = 0
        all_preds = []

        for i in range(0, num_samples, self.batch_size):

            x_batch = X[i:i+self.batch_size].to(self.device)
            y_batch = y[i:i+self.batch_size].to(self.device)

            batch = train_data.clone()
            batch.x = x_batch
            batch.y = y_batch

            self.optimizer.zero_grad()

            logits, _ = self.model(batch)

            loss = self.criterion(logits, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.detach().cpu())

            correct += (preds == y_batch).sum().item()
            total_loss += loss.item() * x_batch.size(0)

        # ✅ Correct metrics
        accuracy = correct / num_samples
        loss = total_loss / num_samples

        # ✅ Correct prediction distribution
        all_preds = torch.cat(all_preds)
        print("Pred dist:", torch.bincount(all_preds).numpy())

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    # ========================================
    # VALIDATION (MINI-BATCH)
    # ========================================
    def validate(self, val_data):

        self.model.eval()

        X = val_data.x
        y = val_data.y

        num_samples = X.size(0)

        total_loss = 0
        correct = 0

        with torch.no_grad():

            for i in range(0, num_samples, self.batch_size):

                x_batch = X[i:i+self.batch_size].to(self.device)
                y_batch = y[i:i+self.batch_size].to(self.device)

                batch = val_data.clone()
                batch.x = x_batch
                batch.y = y_batch

                logits, _ = self.model(batch)

                loss = self.criterion(logits, y_batch)

                preds = torch.argmax(logits, dim=1)

                correct += (preds == y_batch).sum().item()
                total_loss += loss.item() * x_batch.size(0)

        accuracy = correct / num_samples
        loss = total_loss / num_samples

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    # ========================================
    # TRAIN LOOP
    # ========================================
    def train(self, train_data, val_data, num_epochs=None):

        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']

        # ✅ FIX: compute class weights HERE
        counts = torch.bincount(train_data.y)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum()

        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))

        print("Train label dist:", counts.cpu().numpy())
        print("Class weights:", weights.cpu().numpy())

        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print("="*70)

        for epoch in range(1, num_epochs + 1):

            print(f"\n🔥 Epoch {epoch} STARTED")

            train_metrics = self.train_epoch(train_data)
            val_metrics = self.validate(val_data)

            print(f"Epoch {epoch}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Acc:  {train_metrics['accuracy']:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f}")
            print(f"Val Acc:    {val_metrics['accuracy']:.4f}")

            self.scheduler.step(val_metrics['loss'])

        print("\n✅ TRAINING COMPLETE")

        return {'best_val_loss': val_metrics['loss']}

    def load_best_model(self):
        self.checkpoint_manager.load_checkpoint(self.model, self.optimizer)
