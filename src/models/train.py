"""
Training module
"""

import os
from typing import Callable, Any
from dataclasses import dataclass

import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader

@dataclass
class EpochInfo:
    """
    Information from an epoch
    """
    num:int
    loss:float
    acc:float
    #train_acc:float
    #train_loss:float
    #metrics:dict[str, dict[str, float]]
    eval:dict[str, list[Any]]

class Trainer:
    """
    Trainer class
    """

    def __init__(self,
                 criterion:nn.Module,
                 epochs:int,
                 device:str='cpu',
                 no_improve_limit:int=-1,
                 log_epoch:Callable[[EpochInfo],None]|None=None
                 ) -> None:
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.no_improve_limit = no_improve_limit
        if log_epoch is not None:
            self.log_epoch = log_epoch
        else:
            self.log_epoch = lambda epoch: print(f'Train epoch {epoch.num + 1}/{self.epochs}: '
                                                 f'Loss({epoch.loss:6.4f}), '
                                                 f'Accuracy({epoch.acc:6.5f})')

    def evaluate(self,
                  model:nn.Module,
                  loader:DataLoader,
                  epoch:int | None=None,
                  mode:str='validation',
                  verbose:bool=True,
                  advanced:bool=False
                  ) -> tuple[float, float, dict[str, list[Any]]]:
        """
        Evaluate a model

        Args:
            model (Module): the model to evaluate
            loader (DataLoader): the dataloader for the test dataset
            epoch (int | None): the epoch number for printing
            mode (str optional default: 'validation'): mode string for printing
            verbose (bool optional default: True): will not print if False
        Returns:
            A tuple containing the accuracy, loss, and dictionary with test info
            (only populated when using advanced) in that order
        """
        model.eval()
        indices = []
        predicted_labels = []
        true_labels = []
        confidences = []
        total_correct = 0
        total_loss = 0
        # BATCH_SIZE = 64
        total = 0
        for enum, (images, labels) in enumerate(loader):
            print(f'{mode} {f"epoch {epoch + 1}/{self.epochs}" if epoch is not None else ""} '
                  f'({enum + 1}/{len(loader)}): ...\r',
                  end='')
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels = labels.squeeze(1)
            with torch.no_grad():
                outputs, confidence = model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                #If number of samples increase more than one it SHOULD increase as well
                total += images.size(0)
                _, predictions = outputs.max(1)
                total_correct += (labels == predictions).sum()
                if advanced:
                    assert loader.batch_size is not None # dumb assert to please pylance
                    batch_start = enum * loader.batch_size
                    batch_indices = list(range(batch_start, batch_start + images.size(0)))
                    confidences.extend(confidence.cpu().numpy())
                    indices.extend(batch_indices)
                    predicted_labels.extend(predictions.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
        loss = total_loss / total
        accuracy = total_correct / total
        print()

        if verbose:
            print(f'{mode} {f"epoch {epoch + 1}/{self.epochs}" if epoch is not None else ""}: '
                f'Loss({loss:6.4f}), Accuracy({accuracy:6.4f})')

        test_dict = {
            'indices': indices,
            'predicted': predicted_labels,
            'true_labels': true_labels,
            'confidence_score': confidences
        }
        return accuracy, loss, test_dict

    def __call__(self,
                 model:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 train_loader:DataLoader,
                 val_loader:DataLoader,
                 save_dir:str,
                 verbose:bool=True,
                 tensorboard:bool=False,
                 ) -> tuple[list[float], list[float]]:
        accs:list[float] = []
        losses:list[float] = []
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            model.train()
            total:int = 0
            running_loss:float = 0.0
            running_corrects:int = 0
            val_loss:float = 0.0
            best_val_loss:float = float('inf')
            for i, (images, labels) in enumerate(train_loader):
                print(f'Train epoch {epoch + 1}/{self.epochs} ({i + 1}/{len(train_loader)}): ...\r',
                      end='')
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze(1)

                optimizer.zero_grad()
                _ , outputs = model(images)

                loss = self.criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                total += images.size(0)
                _, predictions = outputs.max(1)

                running_loss += loss.item() * images.size(0)
                running_corrects += (predictions == labels).sum()

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total

            print()

            losses.append(epoch_loss)
            accs.append(epoch_acc)
            val_acc, val_loss, eval_dict = self.evaluate(model,
                                               val_loader,
                                               epoch=epoch,
                                               mode='Valid',
                                               verbose=True,
                                               advanced=True)
            self.log_epoch(EpochInfo(epoch,
                                     epoch_loss,
                                     epoch_acc,
                                     eval_dict))
            accs.append(val_acc)
            losses.append(val_loss)

            print(f'{"":-^{50}}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save(model.state_dict(), os.path.join(save_dir, f'test{epoch}.pth'))
            else:
                epochs_no_improve += 1

            if self.no_improve_limit > 0 and epochs_no_improve > self.no_improve_limit:
                print('no further improvement')
                break

        return accs, losses
