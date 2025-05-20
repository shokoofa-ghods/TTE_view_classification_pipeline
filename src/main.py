"""
Main entry point
"""

import sys
import os
import glob

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from matplotlib.axes import Axes

# pylint is unable to resolve the imports but they should be fine
from data import (ALL_DATA_TRANSFORMS,
                  ApplySameTransformToAllFrames,
                  CustomDataset,
                  VALID_LABELS)                 # pylint: disable=import-error
from utils import (visualization,
                   assert_exists,
                   assert_is_file,
                   assert_is_dir)               # pylint: disable=import-error
from models import (ImageCNNLSTM,
                    VideoCNNLSTM,
                    Trainer,
                    EpochInfo)                  # pylint: disable=import-error
from params import get_params                   # pylint: disable=import-error

REQUIRED_PARAMS = ['origin',
                   'train',
                   'eval',
                   'multiframe'
]

def log_tensorboard(epoch:EpochInfo, writer:SummaryWriter, eps:float=1E-9):
    """
    Log an epoch to tensorboard

    Args:
        epoch (EpochInfo): The info about the epoch
        writer (SummaryWriter): The writer to use
        eps (float optional default: 1e-9): Small float to prevent 
            divide by zero
    """
    res = {
        key: {
            'true_pos': 0,
            'false_pos': 0,
            'true_neg': 0,
            'false_neg': 0
        } for key in VALID_LABELS
    }
    # Consider moving to train to save computation
    for i in range(len(epoch.eval['indices'])):
        for label in VALID_LABELS:
            if label == epoch.eval['predicted'][i]:
                if epoch.eval['true_labels'][i] == label:
                    res[label]['true_pos'] += 1
                else:
                    res[label]['false_pos'] += 1
            else:
                if epoch.eval['true_labels'][i] == label:
                    res[label]['false_neg'] += 1
                else:
                    res[label]['true_neg'] += 1
    run_f1 = 0
    for label in VALID_LABELS:
        all_pos = res[label]['true_pos'] + res[label]['false_neg']
        labeled_pos = res[label]['true_pos'] + res[label]['false_pos']
        recall = res[label]['true_pos'] / (all_pos + eps)
        precision = res[label]['true_pos'] / (labeled_pos + eps)
        f1 = 2 * (recall * precision) / (recall + precision + eps)
        run_f1 += f1
        writer.add_scalar(f'Recall/{label}', recall, epoch.num)
        writer.add_scalar(f'Precision/{label}', precision, epoch.num)
        writer.add_scalar(f'F1/{label}', f1, epoch.num)
    writer.add_scalar('Loss', epoch.loss, epoch.num)
    writer.add_scalar('Accuracy', epoch.acc, epoch.num)
    writer.add_scalar('Accuracy', epoch.acc, epoch.num)
    writer.add_scalar('Average_F1', run_f1 / len(VALID_LABELS), epoch.num)

def main() -> int:
    """
    Main function

    Returns:
        The status code to exit with
    """
    # Get program parameters
    try:
        params = get_params(sys.argv[1:], REQUIRED_PARAMS)
    except ValueError as e:
        print(e)
        return 1

    # For non-vital prints
    def log(*args, **kwargs):
        if params.verbose:
            print(*args, **kwargs)

    log('params:', params, sep='\n')

    # Paths to csv files
    train_path = os.path.join(params.original_address, params.train_info_csv)
    val_path = os.path.join(params.original_address, params.val_info_csv)
    test_path = os.path.join(params.original_address, params.test_info_csv)

    # Validation
    try:
        # Check path existence
        assert_exists(params.original_address)
        assert_exists(train_path)
        assert_exists(val_path)
        assert_exists(test_path)
        # Check correct type and permissions
        assert_is_dir(params.original_address)
        assert_is_file(train_path, readable=True)
        assert_is_file(val_path, readable=True)
        assert_is_file(test_path, readable=True)
    except OSError as e:
        print(e)
        return 1

    # Number of all samples per patients
    paths = glob.glob(os.path.join(params.original_address + '/Dataset/**/*/*', '*'))
    log(len(paths))

    # Data Info
    info_train = pd.read_csv(train_path)
    info_val = pd.read_csv(val_path)
    info_test = pd.read_csv(test_path)

    log(info_test.label.unique())

    # Data Distribution
    if params.plot:
        train_dict = info_train['label'].value_counts().to_dict()
        val_dict =info_val['label'].value_counts().to_dict()
        test_dist_dict = info_test['label'].value_counts().to_dict()

        all_classes = set(train_dict.keys()).union(set(val_dict.keys()))
        train_dict_full = {cls: train_dict.get(cls, 0) for cls in all_classes}
        val_dict_full = {cls: val_dict.get(cls, 0) for cls in all_classes}
        test_dict_full = {cls: test_dist_dict.get(cls, 0) for cls in all_classes}

        df_counts = pd.DataFrame({
            'Train': train_dict_full,
            'Validation': val_dict_full,
            'Test': test_dict_full
        })

        df_counts = df_counts.sort_index()

        # df_counts = pd.DataFrame({'Train': train_counts, 'Validation': val_counts})

        ax = df_counts.plot(kind='bar', figsize=(14, 7))
        ax.set_xlabel("Class Name")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Classes in Training and Validation Sets")

        # Show the plot
        plt.show()

    # Datasets
    # TODO: adjust the frame size
    frames = 3 if params.is_multiframe else 1

    subdir = 'multiframe' if params.is_multiframe else 'singleframe'
    save_dir = os.path.join(params.original_address, subdir)

    train_data = CustomDataset(info_train['path'],
                               info_train['label'],
                               ALL_DATA_TRANSFORMS,
                               params.original_address,
                               frames,
                               training_transform=ApplySameTransformToAllFrames(),
                               normalize=False)
    val_data = CustomDataset(info_val['path'],
                             info_val['label'],
                             ALL_DATA_TRANSFORMS,
                             params.original_address,
                             frames)
    # Test visualize
    if params.plot:
        log('Visualizing Samples')
        dt = train_data[10][0][:]
        lb = train_data[10][1]
        num = len(dt)
        visualization(dt[num // 2, -1], lb, save_dir)

        log('Visualizing Samples')
        dt1 = train_data[60][0][:]
        lb1 = train_data[60][1]
        visualization(dt1[num // 2, -1], lb1, save_dir)

    # Data Loaders
    log('Initializing Data Loaders')
    train_loader = DataLoader(train_data,
                              batch_size=32,
                              drop_last=True,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4,
                              persistent_workers=True)
    val_loader = DataLoader(val_data,
                            batch_size=32,
                            drop_last=True,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4,
                            persistent_workers=True)
    if params.verbose:
        log('test read')
        for img, _ in train_loader:
            log(img.shape)
            break

    log('Initializing Model')
    # Model Initialization
    model = VideoCNNLSTM() if params.is_multiframe else ImageCNNLSTM()
    if params.base_model:
        print(f'Loading Model from "{params.base_model}"')
        state_dict = torch.load(os.path.join(params.original_address, params.base_model))
        model.load_state_dict(state_dict)

    # Test model
    if params.verbose:
        model.eval()
        log(model(torch.rand(8, 1, 3, 299, 299)))

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    log(f'Using device: {device}')

    if torch.cuda.device_count() > 1:
        log(f'Let\'s use {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)

    # Training init
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    if params.tensorboard:
        writer = SummaryWriter()
        trainer = Trainer(criterion,
                          params.epochs,
                          device,
                          no_improve_limit=params.loss_limit,
                          log_epoch=lambda e:log_tensorboard(e, writer))
    else:
        trainer = Trainer(criterion,
                          params.epochs,
                          device,
                          no_improve_limit=params.loss_limit)
    
    accs, _losses = trainer(model,
                           optimizer,
                           train_loader,
                           val_loader,
                           save_dir,
                           verbose=params.verbose)

    if params.plot:
        # Generate sample data
        # x = [i for i in range(0,len(accs)//2)]
        # y = [accs[i].cpu() for i in range(0,len(accs),2)]
        # z = [accs[i].cpu() for i in range(1,len(accs),2)]
        x = [i for i in range(len(accs) // 2)]
        y = [accs[i] for i in range(0, len(accs), 2)]
        z = [accs[i] for i in range(1, len(accs), 2)]

        # Create a new figure and set the size
        fig = plt.figure(figsize=(8, 6))

        # Add a new subplot to the figure
        ax:Axes = fig.add_subplot(1, 1, 1)

        # Plot the line graph
        ax.plot(x, y, label='accuracy')
        ax.plot(x, z, label='val_accuracy')
        ax.legend()

        # Set the title and axis labels
        ax.set_title('CNN-LSTM with 16frames with 3 jump-steps, joined psax views')
        ax.set_xlabel('epochs')
        ax.set_ylabel('accuracy')

        # Display the plot
        plt.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())
