from argparse import ArgumentParser
import math
import os
import yaml
import torch
import numpy as np
import random
import json

from model.model import GeoMol
from model.training import train, test, NoamLR
from utils import create_logger, dict_to_str, plot_train_val_loss, save_yaml_file, get_optimizer_and_scheduler
from model.featurization import construct_loader, pdbbind_confs
from model.parsing import parse_train_args, set_hyperparams

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
# torch.multiprocessing.set_sharing_strategy('file_system')

# add training args
args = parse_train_args()

logger = create_logger('train', args.log_dir)
logger.info('Arguments are...')
for arg in vars(args):
    logger.info(f'{arg}: {getattr(args, arg)}')

# seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# def get_pdbbind_loader(args, filename, mode) :
#     pdb_ids_dict = json.load(open(filename))
#     pdb_ids = []
#     for prot_class in pdb_ids_dict :
#         pdb_ids = pdb_ids + pdb_ids_dict[prot_class]
#     print(len(pdb_ids))
#     dataset = pdbbind_confs(args.data_dir, pdb_ids, max_confs=args.n_true_confs)
#     loader = DataLoader(dataset=dataset,
#                         batch_size=args.batch_size,
#                         shuffle=False if mode == 'test' else True,
#                         num_workers=args.num_workers,
#                         pin_memory=False)
#     return loader

# if args.use_egcm :
#     train_loader = get_pdbbind_loader(args, 
#         filename='data/pdbbind_protein_training_set.json', 
#         mode='train')
#     val_loader = get_pdbbind_loader(args,
#         filename='data/pdbbind_protein_test_set.json',
#         mode='test')

def get_pdbbind_loader(args, filename, mode, egcm_dir) :
    with open(filename) as f :
        pdb_ids = [pdb_id.strip() for pdb_id in f.readlines()]
    dataset = pdbbind_confs(args.data_dir, pdb_ids, egcm_dir=egcm_dir, max_confs=args.n_true_confs)
    dataloader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=False if mode == ['val', 'test'] else True,
                        num_workers=args.num_workers,
                        pin_memory=False)
    return dataloader

if args.dataset == 'pdbbind' :
    
    egcm_dir = None
    if args.use_egcm :
        egcm_dir = 'data/egcms/'
    
    train_loader = get_pdbbind_loader(args, 
        filename='data/pdbbind_training_set.txt', 
        mode='train', egcm_dir=egcm_dir)
    val_loader = get_pdbbind_loader(args,
        filename='data/pdbbind_test_set.txt',
        mode='test', egcm_dir=egcm_dir)

else : # regular GeoMol
    train_loader, val_loader = construct_loader(args)
    
print('Loaders done')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build model
if args.restart_dir:
    with open(f'{args.restart_dir}/model_parameters.yml') as f:
        model_parameters = yaml.full_load(f)
    model = GeoMol(**model_parameters).to(device)
    state_dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)

else:
    hyperparams = set_hyperparams(args)
    model_parameters = {'hyperparams': hyperparams,
                        'num_node_features': train_loader.dataset.num_node_features,
                        'num_edge_features': train_loader.dataset.num_edge_features}
    model = GeoMol(**model_parameters).to(device)

print('Model built')

# get optimizer and scheduler
optimizer, scheduler = get_optimizer_and_scheduler(args, model, len(train_loader.dataset))

# record parameters
logger.info(f'\nModel parameters are:\n{dict_to_str(model_parameters)}\n')
yaml_file_name = os.path.join(args.log_dir, 'model_parameters.yml')
save_yaml_file(yaml_file_name, model_parameters)

# instantiate summary writer
writer = SummaryWriter(args.log_dir)

best_val_loss = math.inf
best_epoch = 0

logger.info("Starting training...")
for epoch in range(1, args.n_epochs):
    train_loss = train(model, train_loader, optimizer, device, scheduler, logger if args.verbose else None, epoch, writer)
    logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

    val_loss = test(model, val_loader, device, epoch, writer)
    logger.info("Epoch {}: Validation Loss {}".format(epoch, val_loss))
    if scheduler and not isinstance(scheduler, NoamLR):
        scheduler.step(val_loss)

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pt'))
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, os.path.join(args.log_dir, 'last_model.pt'))

logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))

log_file = os.path.join(args.log_dir, 'train.log')
plot_train_val_loss(log_file)
