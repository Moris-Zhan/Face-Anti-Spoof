import torch
import torch.nn as nn
import numpy as np
import time, os
import math
import logging

# from visdom import Visdom

import models, datasets, utils

import io
from contextlib import redirect_stdout
from torchinfo import summary
from pynvml import *

class Model:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt.ngpu else "cpu")
        
        self.model, self.classifier = models.get_model(opt.net_type, 
                                                       opt.classifier_type, 
                                                       opt.pretrained,
                                                       int(opt.nclasses))
        # -------------------------------------------------------------------------------                                                    
        IM_SHAPE = (self.opt.img_size, self.opt.img_size, 3)
        FEAT_SHAPE = (self.opt.batch_size, 512)
        if self.opt.classifier_type == 'linear' and self.opt.net_type == 'ResNet34DLAS_C':
            FEAT_SHAPE = (self.opt.batch_size, 128)
        elif self.opt.classifier_type == 'linear' and self.opt.net_type == 'IR_50DLAS_C':
            FEAT_SHAPE = (self.opt.batch_size, 128)       
        # -------------------------------------------------------------------------------        
        f = io.StringIO()
        with redirect_stdout(f):        
            summary(self.model, [(self.opt.batch_size, IM_SHAPE[2], IM_SHAPE[0], IM_SHAPE[1])] * 3 )
        lines = f.getvalue()

        with open( os.path.join(self.opt.out_path, "model.txt") ,"w") as f:
            [f.write(line) for line in lines]
        print(lines)
        # ------------------------------------------------------------------------------  
        f = io.StringIO()      
        with redirect_stdout(f):        
            summary(self.classifier, FEAT_SHAPE)
        lines = f.getvalue()

        with open( os.path.join(self.opt.out_path, "classifier.txt") ,"w") as f:
            [f.write(line) for line in lines]
        print(lines)
        # ------------------------------------------------------------------------------
        self.model = self.model.to(self.device)
        self.classifier = self.classifier.to(self.device)

        if opt.ngpu>1:
            self.model = nn.DataParallel(self.model)
            
        self.loss = models.init_loss(opt.loss_type,self.device)
        self.loss = self.loss.to(self.device)

        self.optimizer = utils.get_optimizer(self.model, self.opt)
        self.lr_scheduler = utils.get_lr_scheduler(self.opt, self.optimizer)
        self.alpha_scheduler = utils.get_margin_alpha_scheduler(self.opt)

        self.train_loader = datasets.generate_loader(opt,'train') 
        self.test_loader = datasets.generate_loader(opt,'val')
        
        self.epoch = 0
        self.best_epoch = False
        self.training = False
        self.state = {}
        

        self.train_loss = utils.AverageMeter()
        self.test_loss  = utils.AverageMeter()
        self.batch_time = utils.AverageMeter()
        self.test_metrics = utils.ROCMeter()
        self.best_test_loss = utils.AverageMeter()                    
        self.best_test_loss.update(np.array([np.inf]))

        self.visdom_log_file = os.path.join(self.opt.out_path, 'log_files', 'visdom.log')
        # self.vis = Visdom(port = opt.visdom_port,
        #                   log_to_filename=self.visdom_log_file,
        #                   env=opt.exp_name + '_' + str(opt.fold))
        
        # self.vis_loss_opts = {'xlabel': 'epoch',
        #                       'ylabel': 'loss',
        #                       'title':'losses',
        #                       'legend': ['train_loss', 'val_loss']}
        
        # self.vis_tpr_opts = {'xlabel': 'epoch',
        #                       'ylabel': 'tpr',
        #                       'title':'val_tpr',
        #                       'legend': ['tpr@fpr10-2', 'tpr@fpr10-3', 'tpr@fpr10-4']}
        
        # self.vis_epochloss_opts = {'xlabel': 'epoch',
        #                       'ylabel': 'loss',
        #                       'title':'epoch_losses',
        #                       'legend': ['train_loss', 'val_loss']}

    def train(self):
        
        # Init Log file
        if self.opt.resume:
            logging.info('resuming...\n')
            # Continue training from checkpoint
            self.load_checkpoint()             

        if self.opt.evaluate:
            self.test_loader = datasets.generate_loader(self.opt,'dev_test',True)
            self.test_epoch()
            return

        for epoch in range(self.epoch, self.opt.num_epochs):
            self.epoch = epoch            
            # print(epoch)
            # print(self.opt.num_epochs)
            # freezing model
            if self.opt.freeze_epoch:
                logging.info("freezing model")
                if epoch < self.opt.freeze_epoch:
                    if self.opt.ngpu > 1:
                        for param in self.model.module.parameters():
                            param.requires_grad=False
                    else:
                        for param in self.model.parameters():
                            param.requires_grad=False
                elif epoch == self.opt.freeze_epoch:
                    if self.opt.ngpu > 1:
                        for param in self.model.module.parameters():
                            param.requires_grad=True
                    else:
                        for param in self.model.parameters():
                            param.requires_grad=True
            

            self.lr_scheduler.step()
            self.train_epoch()
            self.test_epoch()
            self.log_epoch()
            #self.vislog_epoch()
            self.create_state()
            self.save_state()  
    
    def train_epoch(self):
        """
        Trains model for 1 epoch
        """
        self.model.train()
        self.classifier.train()
        self.training = True
        torch.set_grad_enabled(self.training)
        self.train_loss.reset()
        self.batch_time.reset()
        time_stamp = time.time()
        self.batch_idx = 0
        for batch_idx, (rgb_data, depth_data, ir_data, target,dir) in enumerate(self.train_loader):
            
            self.batch_idx = batch_idx
            rgb_data = rgb_data.to(self.device) # torch.Size([16, 3, 112, 112])
            depth_data = depth_data.to(self.device)
            ir_data = ir_data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            
            output = self.model(rgb_data, depth_data, ir_data)
            if isinstance(self.classifier, nn.Linear):
                if self.opt.fc:
                    output = self.classifier(output)
            else:
                if self.alpha_scheduler:
                    alpha = self.alpha_scheduler.get_alpha(self.epoch)
                    output = self.classifier(output, target, alpha=alpha)
                else:
                    output = self.classifier(output, target)

            if self.opt.loss_type == 'bce':
                target = target.float()
                loss_tensor = self.loss(output.squeeze(), target)
            else:
                # print(output.size())
                loss_tensor = self.loss(output, target)

            loss_tensor.backward()   

            self.optimizer.step()

            self.train_loss.update(loss_tensor.item())
            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.log_batch(batch_idx)
            # self.vislog_batch(batch_idx)
            
    def test_epoch(self):
        """
        Calculates loss and metrics for test set
        """
        self.training = False
        torch.set_grad_enabled(self.training)
        self.model.eval()
        self.classifier.eval()
        
        self.batch_time.reset()
        self.test_loss.reset()
        self.test_metrics.reset()
        time_stamp = time.time()
        
        for batch_idx, (rgb_data, depth_data, ir_data, target,dir) in enumerate(self.test_loader):
            rgb_data = rgb_data.to(self.device)
            depth_data = depth_data.to(self.device)
            ir_data = ir_data.to(self.device)
            target = target.to(self.device)

            output = self.model(rgb_data, depth_data, ir_data)
            output = self.classifier(output)
            if self.opt.loss_type == 'bce':
                target = target.float()
                loss_tensor = self.loss(output.squeeze(), target)
            else:
                loss_tensor = self.loss(output, target)
            self.test_loss.update(loss_tensor.item())

            if self.opt.loss_type == 'cce' or self.opt.loss_type == 'focal_loss':
                output = torch.nn.functional.softmax(output, dim=1)
            elif self.opt.loss_type == 'bce':
                output = torch.sigmoid(output)

            # soft_output = torch.softmax(output, dim=-1)
            preds = output.to('cpu').detach().numpy()



            for i_batch in range(preds.shape[0]):

                if self.opt.val_save:
                    save_path = 'submission/{fold_n}/'.format(fold_n=self.opt.fold)
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    f = open('submission/{fold_n}/submission10_{exp}.txt'.format(fold_n=self.opt.fold,exp=self.opt.exp), 'a+')
                    depth_dir = dir[i_batch].replace(os.getcwd() + '/data/', '')
                    # rgb_dir = depth_dir.replace('depth','color')
                    # ir_dir = depth_dir.replace('depth','ir')
                    # f.write(rgb_dir + ' ' + depth_dir + ' '+ir_dir+' ' + str(preds[i_batch,1]) +'\n')
                    f.write(depth_dir + ' ' + str(preds[i_batch, 1]) + '\n')


            self.test_metrics.update(target.cpu().numpy(), output.cpu().numpy())

            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.log_batch(batch_idx)
            #self.vislog_batch(batch_idx)
            if self.opt.debug and (batch_idx==10):
                logging.info('Debugging done!')
                break;

        self.best_epoch = self.test_loss.avg < self.best_test_loss.val
        if self.best_epoch:
            # self.best_test_loss.val is container for best loss, 
            # n is not used in the calculation
            self.best_test_loss.update(self.test_loss.avg, n=0)
     
    def calculate_metrics(self, output, target):   
        """
        Calculates test metrix for given batch and its input
        """
        t = target
        o = output
            
        if self.opt.loss_type == 'bce':
            accuracy = (t.byte()==(o>0.5)).float().mean(0).cpu().numpy()  
            batch_result.append(binary_accuracy)
        
        elif self.opt.loss_type == 'cce':
            top1_accuracy = (torch.argmax(o, 1)==t).float().mean().item()
            batch_result.append(top1_accuracy)
        else:
            raise Exception('This loss function is not implemented yet')
                
        return batch_result    
    
    def log_batch(self, batch_idx):
        if batch_idx % self.opt.log_batch_interval == 0:
            cur_len = len(self.train_loader) if self.training else len(self.test_loader)
            cur_loss = self.train_loss if self.training else self.test_loss
            global_step = self.epoch * cur_len + batch_idx
            
            output_string = 'Train ' if self.training else 'Test '
            output_string +='Speed: {:.2f} [samples/sec]: Epoch {}[{:.2f}%]: '.format(self.batch_time.avg,
                                                                                    self.epoch, 100.* batch_idx/cur_len,
                                                                                    )
            
            opt_i_string = 'Global Step: {:d} LearningRate {:.6f}\t'.format(global_step, self.lr_scheduler.get_lr()[0])
            output_string += opt_i_string
            
            loss_i_string = 'Loss: {:.5f}(Avg:{:.5f})\t'.format(cur_loss.val, cur_loss.avg)
            output_string += loss_i_string

            if self.opt.writer is not None and self.training:
                # global_step = self.epoch * cur_len + batch_idx
                self.opt.writer.add_scalar('Step/time_for_end', self.batch_time.avg, global_step)
                self.opt.writer.add_scalar('Step/learning_rate', self.lr_scheduler.get_lr()[0], global_step)
                self.opt.writer.add_scalar('Step/loss', cur_loss.avg, global_step)
                    
            if not self.training:
                output_string+='\n'

                metrics_i_string = 'Accuracy: {:.5f}\t'.format(self.test_metrics.get_accuracy())
                output_string += metrics_i_string
                
            # print(output_string)
            logging.info(output_string)
    
    def vislog_batch(self, batch_idx):
        if batch_idx % self.opt.log_batch_interval == 0:
            loader_len = len(self.train_loader) if self.training else len(self.test_loader)
            cur_loss = self.train_loss if self.training else self.test_loss
            loss_type = 'train_loss' if self.training else 'val_loss'
            
            x_value = self.epoch + batch_idx / loader_len
            y_value = cur_loss.val
            self.vis.line([y_value], [x_value], 
                            name=loss_type, 
                            win='losses', 
                            update='append')
            self.vis.update_window_opts(win='losses', opts=self.vis_loss_opts)
    
    def log_msg(self, msg=''):
        mode = 'a' if msg else 'w'
        if not os.path.exists(os.path.join(self.opt.out_path, 'log_files')):
            os.makedirs(os.path.join(self.opt.out_path, 'log_files'))
        f = open(os.path.join(self.opt.out_path, 'log_files', 'train_log.txt'), mode)
        f.write(msg)
        f.close()
             
    def log_epoch(self):
        """ Epoch results log string"""
        out_train = 'Train: '
        out_test = 'Test:  '
        loss_i_string = 'Loss: {:.5f}\t'.format(self.train_loss.avg)
        out_train += loss_i_string
        loss_i_string = 'Loss: {:.5f}\t'.format(self.test_loss.avg)
        out_test += loss_i_string

        self.opt.writer.add_scalar('Loss/Train', self.train_loss.avg, self.epoch)
        self.opt.writer.add_scalar('Loss/Test', self.test_loss.avg, self.epoch)
            
        out_test+='\nTest:  '
        metrics_i_string = 'TPR@FPR=10-2: {:.4f}\t'.format(self.test_metrics.get_tpr(0.01))
        metrics_i_string += 'TPR@FPR=10-3: {:.4f}\t'.format(self.test_metrics.get_tpr(0.001))
        metrics_i_string += 'TPR@FPR=10-4: {:.4f}\t'.format(self.test_metrics.get_tpr(0.0001))
        out_test += metrics_i_string

        self.opt.writer.add_scalar('Accuracy/TPR@FPR=10-2', self.test_metrics.get_tpr(0.01), self.epoch)
        self.opt.writer.add_scalar('Accuracy/TPR@FPR=10-3', self.test_metrics.get_tpr(0.001), self.epoch)
        self.opt.writer.add_scalar('Accuracy/TPR@FPR=10-4', self.test_metrics.get_tpr(0.0001), self.epoch)
            
        is_best = 'Best ' if self.best_epoch else ''
        out_res = is_best+'Epoch {} results:\n'.format(self.epoch)+out_train+'\n'+out_test+'\n'
        
        logging.info(out_res)

    def vislog_epoch(self):
        x_value = self.epoch
        self.vis.line([self.train_loss.avg], [x_value], 
                        name='train_loss', 
                        win='epoch_losses', 
                        update='append')
        self.vis.line([self.test_loss.avg], [x_value], 
                        name='val_loss', 
                        win='epoch_losses', 
                        update='append')
        self.vis.update_window_opts(win='epoch_losses', opts=self.vis_epochloss_opts)


        self.vis.line([self.test_metrics.get_tpr(0.01)], [x_value], 
                        name='tpr@fpr10-2', 
                        win='val_tpr', 
                        update='append')
        self.vis.line([self.test_metrics.get_tpr(0.001)], [x_value], 
                        name='tpr@fpr10-3', 
                        win='val_tpr', 
                        update='append')
        self.vis.line([self.test_metrics.get_tpr(0.0001)], [x_value], 
                        name='tpr@fpr10-4', 
                        win='val_tpr', 
                        update='append')
        self.vis.update_window_opts(win='val_tpr', opts=self.vis_tpr_opts)
                    
    def create_state(self):
        self.state = {       # Params to be saved in checkpoint
                'epoch' : self.epoch,
                'model_state_dict' : self.model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'best_test_loss' : self.best_test_loss,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }
    
    def save_state(self):
        if self.opt.log_checkpoint == 0:
                self.save_checkpoint('checkpoint.pth')
        else:

            self.save_checkpoint('model_{}.pth'.format(self.epoch))
                  
    def save_checkpoint(self, filename):     # Save model to task_name/checkpoints/filename.pth
        fin_path = os.path.join(self.opt.out_path,'checkpoints', filename)
        torch.save(self.state, fin_path)
        if self.best_epoch:
            best_fin_path = os.path.join(self.opt.out_path, 'checkpoints', 'model_best.pth')
            torch.save(self.state, best_fin_path)
            logging.info("save best epoch {}".format(filename))
           

    def load_checkpoint(self):                            # Load current checkpoint if exists
        fin_path = os.path.join(self.opt.out_path,'checkpoints',self.opt.resume)
        if os.path.isfile(fin_path):
            print("=> loading checkpoint '{}'".format(fin_path))
            checkpoint = torch.load(fin_path, map_location=lambda storage, loc: storage)
            self.epoch = checkpoint['epoch'] + 1
            self.best_test_loss = checkpoint['best_test_loss']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            #self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            logging.info("=> loaded checkpoint '{}' (epoch {})".format(self.opt.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(self.opt.resume))

        # if os.path.isfile(self.visdom_log_file):
        #         self.vis.replay_log(log_filename=self.visdom_log_file)
        #
