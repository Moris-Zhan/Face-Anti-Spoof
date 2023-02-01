import argparse,json,random,os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision as tv

from trainer import Model
from custom import get_opts
import logging
import sys
import importlib
from torch.utils.tensorboard import SummaryWriter

def main():
    
    # Load options
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type=str, default="" # opts.B_pro_4_3
                        ,help = 'Path to config .opt file. Leave blank if loading from opts.py')
    
    conf = parser.parse_args() 
   
    # opt = torch.load(conf.config) if conf.config else get_opts()   
    opt = importlib.import_module(conf.config).get_opts() if conf.config else get_opts()
    
    logging.info('===Options==') 
    d=vars(opt)

    with open('commandline_args.txt', 'w') as f:        
        for key, value in d.items():
            num_space = 25 - len(key)
            try:
                f.write(key + " = " + str(value) + "\n")
            except Exception as e :
                pass

    for key, value in d.items():
        num_space = 25 - len(key)
        try:
            logging.info(": " + key + " " * num_space + str(value))
        except Exception as e:
            print(e)

    # # set yourself train data path
    # # opt.data_root = 'the path for train data and dev data'
    # opt.data_root = '/home/leyan/DataSet/CASIA-CeFA/phase1/'

    # Fix seed
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)
    cudnn.benchmark = True
    
    # Create working directories
    try:
        # os.makedirs(opt.out_path)
        os.makedirs(os.path.join(opt.out_path,'checkpoints'))
        os.makedirs(os.path.join(opt.out_path,'log_files'))
        # opt.writer = SummaryWriter(log_dir=os.path.join(opt.out_path, "tensorboard"))
        logging.info( 'Directory {} was successfully created.'.format(opt.out_path))
                   
    except OSError:
        logging.info( 'Directory {} already exists.'.format(opt.out_path))
        pass
    
    
    # Training
    M = Model(opt)
    M.train()
    '''
    TODO: M.test()
    '''
    
if __name__ == '__main__':
    main()


