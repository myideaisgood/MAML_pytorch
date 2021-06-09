import torch
from torch.utils.data import DataLoader

import numpy as np
import logging
import os
from time import time

from config import parse_args
from model import ModelConvMiniImagenet
from maml import ModelAgnosticMetaLearning
from utils.data_loaders import Dataset
from utils.helpers import *

args = parse_args()


# Few Shot Parameters
N_WAY = args.N_way
K_SHOT = args.K_shot
QUERY_NUM = args.query_num
EVALUATE_TASK = args.evaluate_task

# MAML Parameters
TASK_NUM = args.task_num
NUM_STEPS_TRAIN = args.num_steps_train
NUM_STEPS_TEST = args.num_steps_test
STEP_SIZE = args.step_size
FIRST_ORDER = args.first_order

# Model Parameters
HIDDEN_UNIT = args.hidden_unit

# Session Parameters
GPU_NUM = args.gpu_num
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
META_LR = args.meta_lr
PRINT_EVERY = args.print_every
EVALUATE_EVERY = args.evaluate_every

# Directory Parameters
DATASET = args.dataset
EXP_NAME = args.experiment_name
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
WEIGHTS = args.weights

# Check if directory does not exist
create_path('experiments/')
create_path(EXP_DIR)
create_path(CKPT_DIR)
create_path(LOG_DIR)
create_path(os.path.join(LOG_DIR, 'train'))
create_path(os.path.join(LOG_DIR, 'test'))

# Set up logger
filename = os.path.join(LOG_DIR, 'logs.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

# Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up Dataset
train_dataset = Dataset(args, 'train')
val_dataset = Dataset(args, 'val')
test_dataset = Dataset(args, 'test')

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=TASK_NUM,
    num_workers=2,
    shuffle=True,
    drop_last=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=TASK_NUM,
    num_workers=2,
    shuffle=True,
    drop_last=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=TASK_NUM,
    num_workers=2,
    shuffle=False
)

# Set up model / optimizer
model = ModelConvMiniImagenet(out_features=N_WAY, hidden_size=HIDDEN_UNIT)

meta_optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)

# Set up Loss Functions
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

# Load the pretrained model if exists
init_epoch = 0
best_metrics_val = 0.0
best_metrics_test = 0.0

if os.path.exists(os.path.join(CKPT_DIR, WEIGHTS)):
    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, WEIGHTS))
    checkpoint = torch.load(os.path.join(CKPT_DIR, WEIGHTS))
    init_epoch = checkpoint['epoch_idx']
    best_metrics_val = checkpoint['best_metrics_val']
    best_metrics_test = checkpoint['best_metrics_test']
    model.load_state_dict(checkpoint['model'])
    logging.info('Recover completed. Current epoch = #%d, best metrics (val) = %.3f, best metrics (test) = %.3f' % (init_epoch, best_metrics_val, best_metrics_test))

# Set up meta learner
metalearner = ModelAgnosticMetaLearning(model,
                                        meta_optimizer,
                                        first_order=FIRST_ORDER,
                                        num_adaptation_steps=NUM_STEPS_TRAIN,
                                        step_size=STEP_SIZE,
                                        loss_function=criterion,
                                        device=device)

for epoch_idx in range(init_epoch+1, EPOCHS):

    metalearner.set_adaptation_step(NUM_STEPS_TRAIN)
    metalearner.train(train_dataloader, max_batches=BATCH_SIZE, verbose=True, desc='Training', leave=False)
    
    if epoch_idx % EVALUATE_EVERY == 0:
        metalearner.set_adaptation_step(NUM_STEPS_TEST)
        results_val = metalearner.evaluate(val_dataloader, max_batches=EVALUATE_TASK, verbose=True)
        results_test = metalearner.evaluate(test_dataloader, max_batches=EVALUATE_TASK, verbose=True)

        loss_val = results_val['mean_outer_loss']
        acc_val = results_val['accuracies_after']

        loss_test = results_test['mean_outer_loss']
        acc_test = results_test['accuracies_after']

        # Save model if best metric
        if acc_val >= best_metrics_val:
            output_path = os.path.join(CKPT_DIR, WEIGHTS)
            best_metrics_val = acc_val
            best_metrics_test = acc_test

            torch.save({
                'epoch_idx': epoch_idx,
                'best_metrics_val': best_metrics_val,
                'best_metrics_test': best_metrics_test,
                'model': model.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)
        
        logging.info('Val  [Epoch %d/%d] Loss = %.4f  Accuracy = %.4f' %(epoch_idx, EPOCHS, loss_val, acc_val))
        logging.info('Test [Epoch %d/%d] Loss = %.4f  Accuracy = %.4f' %(epoch_idx, EPOCHS, loss_test, acc_test))