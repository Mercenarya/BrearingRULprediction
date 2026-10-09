"""
Trainer class for handling model training, validation, and checkpointing.
"""
import os, sys

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT, "..", ".."))
sys.path.append(ROOT)

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb
from tqdm import tqdm
from src.evaluation.metrics import calculate_phm_score

class Trainer:
    """
    Trainer class for RUL prediction models.
    """

    def __init__(self, model, criterion, optimizer, scheduler=None, device='cuda',
                 early_stopping_patience=10, checkpoint_dir='checkpoints',
                 use_wandb=False):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): Model to train
            criterion (nn.Module): Loss function
            optimizer (torch.optim.Optimizer): Optimizer
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler
            device (str): Device to use for training ('cpu' or 'cuda')
            early_stopping_patience (int): Number of epochs to wait for improvement
            checkpoint_dir (str): Directory to save checkpoints
            use_wandb (bool): Whether to log to Weights & Biases
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader, epoch_idx=0, num_epochs=1):
        """
        Train for one epoch with progress bar.

        Args:
            train_loader (DataLoader): Training data loader
            epoch_idx (int): Current epoch index
            num_epochs (int): Total number of epochs

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        # Thêm tqdm hiển thị tiến trình từng Batch
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch_idx+1}/{num_epochs} [Train]",
            leave=False
        )

        for batch in pbar:
            # Get data
            vib_inputs = batch['vibration'].to(self.device)
            temp_inputs = batch['temperature'].to(self.device)
            labels = batch['rul'].to(self.device).unsqueeze(1)  # Ensure correct shape

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(vib_inputs, temp_inputs)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Cập nhật giá trị loss thời gian thực trên thanh tiến trình
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / num_batches
        return epoch_loss

    def validate(self, val_loader):
        """
        Validate the model with progress bar.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            tuple: (val_loss, val_metrics) where val_metrics is a dict of metrics
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # Thêm tqdm cho lượt validation
        pbar = tqdm(val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for batch in pbar:
                # Get data
                vib_inputs = batch['vibration'].to(self.device)
                temp_inputs = batch['temperature'].to(self.device)
                labels = batch['rul'].to(self.device).unsqueeze(1)

                # Forward pass
                outputs = self.model(vib_inputs, temp_inputs)

                # Calculate loss
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                # Store predictions and labels for metrics
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = running_loss / len(val_loader)

        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        r2 = r2_score(all_labels, all_preds)

        # Calculate PHM score
        phm_score = calculate_phm_score(all_labels.flatten(), all_preds.flatten())

        val_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'phm_score': phm_score
        }

        return val_loss, val_metrics

    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.

        Args:
            test_loader (DataLoader): Test data loader

        Returns:
            tuple: (test_loss, test_metrics)
        """
        return self.validate(test_loader)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch number
            val_loss (float): Validation loss
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved new best model to {best_path}", flush=True)

    def train(self, train_loader, val_loader, num_epochs):
        """
        Main training loop.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...", flush=True)

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader, epoch, num_epochs)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_metrics['mae'],
                    'val_rmse': val_metrics['rmse'],
                    'val_r2': val_metrics['r2'],
                    'val_phm_score': val_metrics['phm_score']
                })

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val MAE: {val_metrics['mae']:.4f}", flush=True)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.epochs_no_improve += 1

            # Early stopping check
            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs", flush=True)
                break

        print("Training completed!", flush=True)

if __name__ == "__main__":
    # Example usage
    print("Trainer class ready")