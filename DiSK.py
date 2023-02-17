import sys, time, torch, random, argparse, json, math, shutil
import itertools
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.distributions import Categorical

from typing import Type, Any, Callable, Union, List, Optional
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path


from utils import get_model_infos
from config_utils import load_config
from log_utils import AverageMeter, ProgressMeter, time_string, convert_secs2time
from starts import prepare_logger
from get_dataset_with_transform import get_datasets


def m__get_prefix( args ):
    use_kl = args.use_kl
    use_aux = args.use_auxiliary
    use_oracle = args.use_oracle
    use_prob = args.use_prob

    if (use_kl == 1) and (use_aux == 1) and (use_oracle==0) and (use_prob==0):
        ## Training standard KD
        prefix = 'disk-KD-' \
            + args.dataset +  '-' \
            + args.model_name +  '-' \
            + args.global_model_name + '-' \
            + str(args.use_kl)   + '-' \
            + str(args.use_prob)  + '-' \
            + str(args.use_oracle)  + '-' \
            + str(args.use_auxiliary)  + '-'  \
            + str(args.s_lr) + '-' \
            + str(args.g_lr) + '-' \
            + str(args.base_opt_type) + '-' \
            + str(args.wd) + '-' \
            + str(args.KD_alpha) + '-' \
            + str(args.KD_temperature) + '-' \
            + str(args.l_aux) + '-' \
            + str(args.l_kl) + '-' \
            + str(args.shuffle) + '-' \
            + str(args.epochs) + '-' 
    elif (use_kl == 0) and (use_aux == 1) and (use_oracle==0) and (use_prob==0):
        ## Training standard CE
        prefix = 'disk-CE-' \
            + args.dataset +  '-' \
            + args.model_name +  '-' \
            + str(args.use_kl)   + '-' \
            + str(args.use_prob)  + '-' \
            + str(args.use_oracle)  + '-' \
            + str(args.use_auxiliary)  + '-'  \
            + str(args.s_lr) + '-' \
            + str(args.g_lr) + '-' \
            + str(args.base_opt_type) + '-' \
            + str(args.wd) + '-' \
            + str(args.KD_alpha) + '-' \
            + str(args.KD_temperature) + '-' \
            + str(args.l_aux) + '-' \
            + str(args.l_kl) + '-' \
            + str(args.shuffle) + '-' \
            + str(args.epochs) + '-' 
    else:
        prefix = 'disk-hints10-' \
            + args.dataset +  '-' \
            + args.model_name +  '-' \
            + args.routing_name +  '-' \
            + args.global_model_name + '-' \
            + str(args.base_strategy) + '-'  \
            + str(args.penalty)  + '-'  \
            + str(args.use_kl)   + '-' \
            + str(args.use_prob)  + '-' \
            + str(args.use_oracle)  + '-' \
            + str(args.use_auxiliary)  + '-'  \
            + str(args.use_only_bn) + '-'  \
            + str(args.use_alt_min) + '-'  \
            + str(args.s_lr) + '-' \
            + str(args.g_lr) + '-' \
            + str(args.wd) + '-' \
            + str(args.base_opt_type) + '-' \
            + str(args.routing_opt_type) + '-' \
            + str(args.base_method) + '-' \
            + str(args.KD_alpha) + '-' \
            + str(args.KD_temperature) + '-' \
            + str(args.Ti) + '-' \
            + str(args.budget_Ti) + '-' \
            + str(args.budget_g_min) + '-' \
            + str(args.budget_g_max) + '-' \
            + str(args.lmbda_min) + '-' \
            + str(args.lmbda) + '-' \
            + str(args.b_add_sparsity_alt_min) + '-' \
            + str(args.l_aux) + '-' \
            + str(args.l_kl) + '-' \
            + str(args.l_oracle) + '-' \
            + str(args.l_nll) + '-' \
            + str(args.epochs) + '-' 
    return prefix


def get_model_prefix( args ):
    prefix = './models/' + m__get_prefix(args) 
    return prefix

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+'model_best.pth.tar')

def update_lmbda_climb_val( args ):
    lmbda = args.lmbda  
    l_min = args.lmbda_min
    l_max = args.lmbda #_max

    lmbda_step = (l_max - l_min) / args.Ti
    if not args.lmbda_climb: lmbda_step = -1 * lmbda_step
    lmbda = args.lmbda_climb_val + lmbda_step

    args.lmbda_climb_val = max( lmbda, args.lmbda_min )

def get_lmbda_val( args, gate, y_one_hot, s_y_hat, t_y_hat, y, epoch   ):
    lmbda = args.lmbda  
    l_min = args.lmbda_min
    l_max = args.lmbda #_max

    if args.lmbda_adaptive == 1:
        lmbda = args.lmbda * (epoch+1)
    elif args.lmbda_adaptive == 2:
        lmbda = args.lmbda / (epoch+1)
    elif args.lmbda_adaptive == 3:
        lmbda = args.lmbda * ( (epoch+1) ** 0.5)
    elif args.lmbda_adaptive == 4:
        lmbda = l_min + ( (l_max - l_min) / args.epochs ) * epoch
    elif args.lmbda_adaptive == 5:
        Ti = args.Ti #args.epochs
        T = epoch % Ti #args.epochs
        lmbda = l_min + (l_max - l_min) * (1 - math.cos(math.pi * T / Ti) ) / 2
    elif args.lmbda_adaptive == 6:
        lmbda = args.lmbda_dual
    elif args.lmbda_adaptive == 7:
        lmbda = args.lmbda_climb_val

    return lmbda

def get_reg(  args, gate, y_one_hot, s_y_hat, t_y_hat, y, epoch ):
    dec=4
    lmbda = get_lmbda_val( args, gate, y_one_hot, s_y_hat, t_y_hat, y, epoch )

    if args.penalty == 24:

        #b_correct = (y != torch.argmax(s_y_hat, dim=1)) * 1. 
        #b_correct = F.cross_entropy( s_y_hat, y, reduction='none' )
        b_correct = F.cross_entropy( s_y_hat,  torch.argmax(t_y_hat, dim=1) , reduction='none' )
        b_correct = torch.clamp( b_correct, max=args.max_ce )  

        #z = F.softmax( s_y_hat, dim=1 )
        #z = z + 0.00001
        #b_correct = F.nll_loss( torch.log(z), y, reduction='none' )
        #b_correct = torch.clamp(b_correct, max=1.)

        b_correct = b_correct.view(-1, 1)
        #n_correct = torch.sum(b_correct).detach()
        n_correct = torch.sum( (y != torch.argmax(s_y_hat, dim=1)) * 1. )

        #reg = F.relu( (1/n_correct) * torch.sum( b_correct ) - args.budget_g )

        #reg = F.relu( (1/n_correct) * torch.sum( gate * b_correct ) - args.budget_g )
        #loss = lmbda * reg + 0.5 * args.rho * reg * reg + torch.mean(  torch.log( 1 + (args.class_num-1) * gate ) ) 
        #loss = lmbda * reg + 0.5 * args.rho * reg * reg + torch.mean( 0.1 *  torch.log( 1 + (args.class_num-1) * gate ) ) 

        #reg = F.relu( torch.mean( gate * b_correct ) - args.budget_g )
        reg = F.relu( (1/n_correct) * torch.sum( gate * b_correct ) - args.budget_g )
        #reg = torch.abs( (1/n_correct) * torch.sum( gate * b_correct ) - args.budget_g )
        #loss = lmbda * reg #+ 0.5 * args.rho * reg * reg + torch.mean( 0.1 * torch.log( 1 + (args.class_num-1) * gate ) ) 
        loss = lmbda * reg + torch.mean( args.l_nll * torch.log( 1 + args.topK * gate ) ) 

        verbose=True
        if verbose and args.epoch_i==0:
            args.logger.log(  '        budget ' + str( round(args.budget_g, dec) ) +  ' -- budget_max ' + str( round(args.budget_g_max, dec) ) +  
                    ' torch.sum( b!=y ) ' + str( round(  torch.sum( (y != torch.argmax(s_y_hat, dim=1)) * 1.  ).item(), dec) )  +
                    ' CE-min ' + str( round(torch.min( b_correct ).item(), dec) ) + ' CE-max ' + str( round(torch.max( b_correct ).item(), dec) )  +
                    '  torch.sum( gate * b_correct ) ' + str( round( ((1/n_correct) * torch.sum( gate * b_correct )).item(), dec) ) + 
                    '  torch.sum( gate * b_correct ) - n_correct * args.budget_g ' + str( round(((1/n_correct) * torch.sum( gate * b_correct ) - args.budget_g).item(), dec) ) +
                    ' lmbda * reg ' + str( round((lmbda * reg ).item(), dec) ) +
                    ' 0.5 * args.rho * reg * reg ' + str(round( ( 0.5 * args.rho * reg * reg ).item(), dec)) + 
                    ' sparsity-loss=' + str( round( loss.item(), dec ) ) +  
                    ' lmbda ' + str( round(lmbda, dec) ) + ' -- sum(g) ' + str(round(torch.sum(gate).item(), dec)) +  
                    ' n_correct = ' + str( n_correct ) +  ' -- normalizer = ' + str( round(torch.mean( args.l_nll * torch.log( 1 + (args.topK) * gate ) ).item(), dec) ) )

        #            ' clf-loss=' + str( round(clf_loss, dec) ) +  ' sparsity-loss=' + str( round( loss.item(), dec ) ) +  
        #            ' n_correct = ' + str( n_correct ) +  ' -- normalizer = ' + str( round(torch.mean(  torch.log( 1 + (args.class_num-1) * gate ) ).item(), dec) ) )


    elif args.penalty == 25:

        #b_correct = (y != torch.argmax(s_y_hat, dim=1)) * 1. 
        b_correct = F.cross_entropy( s_y_hat, y, reduction='none' )
        #b_correct = F.cross_entropy( s_y_hat,  torch.argmax(t_y_hat, dim=1) , reduction='none' )
        b_correct = torch.clamp( b_correct, max=args.max_ce )  

        #z = F.softmax( s_y_hat, dim=1 )
        #z = z + 0.00001
        #b_correct = F.nll_loss( torch.log(z), y, reduction='none' )
        #b_correct = torch.clamp(b_correct, max=1.)

        b_correct = b_correct.view(-1, 1)
        #n_correct = torch.sum(b_correct).detach()
        n_correct = torch.sum( (y != torch.argmax(s_y_hat, dim=1)) * 1. )

        #reg = F.relu( (1/n_correct) * torch.sum( b_correct ) - args.budget_g )

        #reg = F.relu( (1/n_correct) * torch.sum( gate * b_correct ) - args.budget_g )
        #loss = lmbda * reg + 0.5 * args.rho * reg * reg + torch.mean(  torch.log( 1 + (args.class_num-1) * gate ) ) 
        #loss = lmbda * reg + 0.5 * args.rho * reg * reg + torch.mean( 0.1 *  torch.log( 1 + (args.class_num-1) * gate ) ) 

        #reg = F.relu( torch.mean( gate * b_correct ) - args.budget_g )
        reg = F.relu( (1/n_correct) * torch.sum( gate * b_correct ) - args.budget_g )
        #reg = torch.abs( (1/n_correct) * torch.sum( gate * b_correct ) - args.budget_g )
        #loss = lmbda * reg #+ 0.5 * args.rho * reg * reg + torch.mean( 0.1 * torch.log( 1 + (args.class_num-1) * gate ) ) 
        loss = lmbda * reg #+ torch.mean( args.l_nll * torch.log( 1 + args.topK * gate ) ) 

        verbose=True
        if verbose and args.epoch_i==0:
            args.logger.log(  '        budget ' + str( round(args.budget_g, dec) ) +  ' -- budget_max ' + str( round(args.budget_g_max, dec) ) +  
                    ' torch.sum( b!=y ) ' + str( round(  torch.sum( (y != torch.argmax(s_y_hat, dim=1)) * 1.  ).item(), dec) )  +
                    ' CE-min ' + str( round(torch.min( b_correct ).item(), dec) ) + ' CE-max ' + str( round(torch.max( b_correct ).item(), dec) )  +
                    '  torch.sum( gate * b_correct ) ' + str( round( ((1/n_correct) * torch.sum( gate * b_correct )).item(), dec) ) + 
                    '  torch.sum( gate * b_correct ) - n_correct * args.budget_g ' + str( round(((1/n_correct) * torch.sum( gate * b_correct ) - args.budget_g).item(), dec) ) +
                    ' lmbda * reg ' + str( round((lmbda * reg ).item(), dec) ) +
                    ' 0.5 * args.rho * reg * reg ' + str(round( ( 0.5 * args.rho * reg * reg ).item(), dec)) + 
                    ' sparsity-loss=' + str( round( loss.item(), dec ) ) +  
                    ' lmbda ' + str( round(lmbda, dec) ) + ' -- sum(g) ' + str(round(torch.sum(gate).item(), dec)) +  
                    ' n_correct = ' + str( n_correct ) +  ' -- normalizer = ' + str( round(torch.mean( args.l_nll * torch.log( 1 + (args.topK) * gate ) ).item(), dec) ) )

        #            ' clf-loss=' + str( round(clf_loss, dec) ) +  ' sparsity-loss=' + str( round( loss.item(), dec ) ) +  
        #            ' n_correct = ' + str( n_correct ) +  ' -- normalizer = ' + str( round(torch.mean(  torch.log( 1 + (args.class_num-1) * gate ) ).item(), dec) ) )


    else:
        print('Penalty Loss undefined : ', args.penalty)
        assert(1==2)

    return reg, loss

def get_updated_budget_g( args, epoch ):
    Ti = args.budget_Ti #args.epochs
    T = epoch % Ti #args.epochs

    if args.primal_budget_update == 1:
        pass
    else:
        args.budget_g = args.budget_g_max + (args.budget_g_min - args.budget_g_max) * (1 - math.cos(math.pi * T / Ti) ) / 2

def get_penalty_loss( args, gate, y_one_hot, s_y_hat, t_y_hat, y, epoch ):        
    get_updated_budget_g( args, epoch )
    reg, loss = get_reg(  args, gate, y_one_hot, s_y_hat, t_y_hat, y, epoch )

    if reg.item()>0:
        args.lmbda_climb = True
    else:
        args.lmbda_climb = False

    return loss    

def primal_budget_update_step( logger, args, train_loader, gating, teacher, student, epoch=1, grad_g_so_far=1000., abs_loss_diff_so_far=1000. ):
    logger.log('primal_budget_update_step')

    teacher.eval()
    gating.eval()
    student.eval()

    gamma = 0.02
    grad = 0.0
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            x = images 
            y = target 

            t_ft, t_logits, t_all_ft = teacher(x)
            s_ft, s_logits, s_all_ft = student(x)
            gate, gate_logits = gating( s_ft, s_logits, t_ft, t_logits, y, s_all_ft, t_all_ft  )

            s_y_hat = s_logits
            t_y_hat = t_logits
            y_one_hot = F.one_hot( y, num_classes=args.class_num )

            reg, _ = get_reg(  args, gate, y_one_hot, s_y_hat, t_y_hat, y, epoch )
            reg = args.lmbda_dual * (reg > 0.0) + args.rho * reg

            grad += gamma * reg

    grad /= len(train_loader)

    #grad = grad.item()
    old_lmbda_dual = args.budget_g

    #if ( grad_g_so_far < 0.001 ) and (abs_loss_diff_so_far < 0.1):
    #if ( grad_g_so_far < 0.001 ) and (abs_loss_diff_so_far > 5.):
    if (abs_loss_diff_so_far > 5.):
        args.budget_g -= grad
        args.budget_g = max( args.budget_g_min, args.budget_g.item() )

        logger.log(' -- old epsilon=' + str(old_lmbda_dual) +  ' -- new epsilon= ' + str(args.budget_g) + ' -- grad ' + str(grad) )
        return True

    return False


def get_grad_with_names(model, old_val, new_val):
    mean_direction = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            mean_direction += torch.sum( param.grad * ( new_val[name] - old_val[name] ) )
    return mean_direction

def dual_update_step_correction_term( old_s, old_g, new_s, new_g, opt_s, opt_g, logger, args, train_loader, gating, teacher, student, epoch=1 ):
    logger.log('dual_update_step')
    model_s, model_g = student, gating

    teacher.eval()
    gating.eval()
    student.eval()

    grad = 0.0
    grad_s, grad_g = 0.0, 0.0
    for i, (images, target) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            x = images 
            y = target 

            opt_s.zero_grad()
            opt_g.zero_grad()

            with torch.no_grad():
                t_ft, t_logits, t_all_ft = teacher(x)

            s_ft, s_logits, s_all_ft = student(x)
            gate, gate_logits = gating( s_ft, s_logits, t_ft, t_logits, y, s_all_ft, t_all_ft  )

            s_y_hat = s_logits
            t_y_hat = t_logits
            y_one_hot = F.one_hot( y, num_classes=args.class_num )

            reg, _ = get_reg(  args, gate, y_one_hot, s_y_hat, t_y_hat, y, epoch )
            grad += args.rho * reg.item()

            reg.backward()
            grad_s += get_grad_with_names(model_s, old_s, new_s)
            grad_g += get_grad_with_names(model_g, old_g, new_g)

            
    N = len(train_loader)
    grad /= N
    grad_s /= N
    grad_g /= N

    grad = grad + grad_s + grad_g
    grad = grad.item()
    
    old_lmbda_dual = args.lmbda_dual
    tau = 1. 
    omega = 4.

    epoch = epoch % args.steps
    #if epoch < args.steps//args.T:
    if epoch < 0:
        args.lmbda_dual = (1/(1+tau)) * ( tau * args.lmbda_dual - (grad / omega) )
    else:    
        args.lmbda_dual = args.lmbda_dual + grad 
    #args.lmbda_dual = max( args.lmbda_dual, args.lmbda_min )    
    args.lmbda_dual = max( args.lmbda_dual, 0.0 )    

    logger.log( ' old -- ' + str(old_lmbda_dual) + ' -- new ' + str(args.lmbda_dual) + ' -- grad ' + str(grad) )



def dual_update_step( logger, args, train_loader, gating, teacher, student, epoch=1 ):
    logger.log('dual_update_step')

    teacher.eval()
    gating.eval()
    student.eval()

    grad = 0.0
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            x = images 
            y = target 

            t_ft, t_logits, t_all_ft = teacher(x)
            s_ft, s_logits, s_all_ft = student(x)
            gate, gate_logits = gating( s_ft, s_logits, t_ft, t_logits, y, s_all_ft, t_all_ft  )

            s_y_hat = s_logits
            t_y_hat = t_logits
            y_one_hot = F.one_hot( y, num_classes=args.class_num )

            reg, _ = get_reg(  args, gate, y_one_hot, s_y_hat, t_y_hat, y, epoch )
            grad += args.rho * reg.item()

    grad /= len(train_loader)
    old_lmbda_dual = args.lmbda_dual
    tau = 1. 
    omega = 4.

    epoch = epoch % args.steps
    if epoch < args.steps//args.T:
    #if epoch < 0:
        args.lmbda_dual = (1/(1+tau)) * ( tau * args.lmbda_dual - (grad / omega) )
    else:    
        args.lmbda_dual = args.lmbda_dual + grad
    args.lmbda_dual = max( args.lmbda_dual, args.lmbda_min )    

    logger.log( ' old -- ' + str(old_lmbda_dual) + ' -- new ' + str(args.lmbda_dual) + ' -- grad ' + str(grad) )

def get_nll_loss_with_hint( args, s_logits, t_logits, s_pred, t_pred, target, gate, y_one_hot, temperature ):

        if args.use_prob == 42:

            topk, indices = torch.topk( t_logits, args.topK ) 
            one_hot = torch.sum( F.one_hot( indices, num_classes=args.class_num ), dim=1 )
            one_hot = torch.max( one_hot, y_one_hot )


            #print( '33 ... ' )
            z = (s_logits / args.KD_temperature_s) #* one_hot
            z = F.softmax( z, dim=1 )

            q = gate.view(-1, 1) #gate_logits #/ temperature
            #q = F.sigmoid(gate_logits) #F.softmax( q, dim=1 )

            ty = (t_logits / temperature) * one_hot
            ty = F.softmax( ty, dim=1 )

            #print('one_hot -- ', one_hot[0] )

            #z = z + ( y_one_hot * q * (1. - ty) )
            #z = (1-q) * z + q * ty #* y_one_hot 
            z = z + q * one_hot

            #log_student = F.log_softmax(q / temperature, dim=1)
            sof_teacher = ty #F.softmax(t_logits / temperature, dim=1)
            sof_teacher = sof_teacher * one_hot
            #KD_loss = F.kl_div(log_student, sof_teacher, reduction="batchmean") * (
            #    args.l_nll * temperature * temperature
            #)

            N = z.size(0)
            KD_loss = args.l_nll * args.KD_temperature_s * temperature * torch.sum( - sof_teacher * torch.log(z) ) / N

            z = q.detach() 
            weights = torch.sum( y_one_hot * (q < 0.5) * 1., dim=1 ) 

        elif args.use_prob == 50:

            topk, indices = torch.topk( t_logits, args.topK ) 
            one_hot = torch.sum( F.one_hot( indices, num_classes=args.class_num ), dim=1 )
            one_hot = torch.max( one_hot, y_one_hot )


            #print( '33 ... ' )
            z = (s_logits / args.KD_temperature_s) #* one_hot
            z = F.softmax( z, dim=1 )

            q = gate.view(-1, 1) #gate_logits #/ temperature
            #q = F.sigmoid(gate_logits) #F.softmax( q, dim=1 )

            min_vals, _ = torch.min(t_logits, 1, keepdim=True)
            
            '''
            print(' 1-one_hot ', (1-one_hot).size())
            print(' min_vals ', min_vals.size())
            tmp = (t_logits / temperature) * one_hot + (1-one_hot)* min_vals

            print('t_logits -- ', t_logits[0])
            print('min t_logits -- ', min_vals[0])
            print('tmp  -- ', tmp[0])
            print('KD -- ', F.softmax( t_logits/temperature , dim=1)[0])
            print('Old DiSK -- ', F.softmax( t_logits/temperature * one_hot , dim=1)[0])
            print('NewDiSK -- ', F.softmax( t_logits/temperature * one_hot  + (1-one_hot)* min_vals, dim=1)[0])
            exit(1)

            ty = (t_logits / temperature) * one_hot
            '''
            ty = (t_logits / temperature) * one_hot + min_vals * (1-one_hot)  
            ty = F.softmax( ty, dim=1 )

            #print('one_hot -- ', one_hot[0] )

            #z = z + ( y_one_hot * q * (1. - ty) )
            #z = (1-q) * z + q * ty #* y_one_hot 
            z = z + q * one_hot

            #log_student = F.log_softmax(q / temperature, dim=1)
            sof_teacher = ty #F.softmax(t_logits / temperature, dim=1)
            sof_teacher = sof_teacher * one_hot
            #KD_loss = F.kl_div(log_student, sof_teacher, reduction="batchmean") * (
            #    args.l_nll * temperature * temperature
            #)

            N = z.size(0)
            KD_loss = args.l_nll * args.KD_temperature_s * temperature * torch.sum( - sof_teacher * torch.log(z) ) / N

            z = q.detach() 
            weights = torch.sum( y_one_hot * (q < 0.5) * 1., dim=1 ) 


        nll_loss = KD_loss
        return nll_loss, weights #, z, sg_gate


def get_loss( logger, g_data, data, target, model, global_model, gating_model, criterion, args, part, add_gate_loss=False, batch_idx=0, vec_stats=None, vec_s_pred=None, vec_t_pred=None, learn_gate=False, epoch=1  ):
    assert( args.base_method == 'hybrid_kd_inst' )

    y_one_hot = F.one_hot( target, num_classes=gating_model.n_labels )

    with torch.no_grad():
        t_ft, t_logits, t_all_ft = global_model(g_data)

    if part=='gating':
        with torch.no_grad():
            s_ft, s_logits, s_all_ft = model(data)
        gate, gate_logits = gating_model( s_ft, s_logits, t_ft, t_logits, target, s_all_ft, t_all_ft  )
    elif part=='student':
        s_ft, s_logits, s_all_ft = model(data)
        if args.use_only_bn == 1:
           assert(1==2)
        else:
           if args.use_alt_min == 0:
               gate, gate_logits = gating_model( s_ft, s_logits, t_ft, t_logits, target, s_all_ft, t_all_ft  )
           else:    
               with torch.no_grad():
                   gate, gate_logits = gating_model( s_ft, s_logits, t_ft, t_logits, target, s_all_ft, t_all_ft )

    t_pred = torch.argmax( t_logits, dim=1 )
    s_pred = torch.argmax( s_logits, dim=1 )

    # Penalty to ensure G(x) being sparse
    loss = get_penalty_loss( args, gate, y_one_hot, s_logits, t_logits, target, epoch )
    penalty_loss = loss.clone().detach()

    if part=='student' and args.use_alt_min==1 and args.b_add_sparsity_alt_min==0:
        penalty_loss = 0. * loss
        loss = 0. * loss

    # Auxiliary loss function
    aux_loss = (0. * loss).clone().detach()
    if args.use_auxiliary == 1:
        aux_loss = args.l_aux * F.cross_entropy(s_logits, target)
        loss += aux_loss 

    # Add KL Div if asked
    temperature = args.KD_temperature
    if args.use_kl == 1:
        log_student = F.log_softmax(s_logits / temperature, dim=1)
        sof_teacher = F.softmax(t_logits / temperature, dim=1)
        KD_loss = F.kl_div(log_student, sof_teacher, reduction="batchmean") * (
            args.l_kl * temperature * temperature
        )
        loss += KD_loss 

    # Add  (1-g) CE(y, y_hat) if asked
    wt_loss = 0. * loss
    if args.use_oracle != 0: assert(1==2)

    # NLL Loss with hints something like -y log( max(b(x), g(x)) )
    if args.use_prob != 0:
        nll_loss, weights = get_nll_loss_with_hint( args, s_logits, t_logits, s_pred, t_pred, target, gate, y_one_hot, temperature )
        loss += nll_loss 

        wt_loss = wt_loss.clone().detach() + nll_loss.clone().detach()

    sc_tic   = torch.mean( 1. * (t_pred != target) * (s_pred == target) )
    sic_tc   = torch.mean( 1. * (t_pred == target) * (s_pred != target) )
    sc_tc    = torch.mean( 1. * (t_pred == target) * (s_pred == target) )
    sic_tic  = torch.mean( 1. * (t_pred != target) * (s_pred != target) )

    return loss, s_logits, t_logits, s_pred, t_pred, penalty_loss, wt_loss, torch.mean(weights), aux_loss, sc_tic, sic_tc, sc_tc, sic_tic

def train_part(logger, train_loader, model, global_model, routingNet, criterion, optimizer, epoch, args, s_optimizer, t_optimizer, part='student'):
    assert( args.base_method == 'hybrid_kd_inst' )

    if args.base_method in ['ce', 'kd']: #, 'hybrid_kd_inst']:
        PARTS = ['student']

    if args.base_method in ['hybrid_kd_inst']:
        #PARTS = ['gating', 'student']
        PARTS = [part]

    logger.log(str(PARTS))
    for part in PARTS: 
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        inst_losses = AverageMeter('Inst', ':.4e')
        wt_losses = AverageMeter('Wt', ':.4e')
        aux_losses = AverageMeter('Aux', ':.4e')
        gt_losses = AverageMeter('Gt', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        Ttop1 = AverageMeter('TAcc@1', ':6.2f')
        Ttop5 = AverageMeter('TAcc@5', ':6.2f')

        progress = ProgressMeter(
            logger,
            len(train_loader),
            [batch_time, data_time, losses, top1, Ttop1, inst_losses, wt_losses, aux_losses, gt_losses],
            prefix="[{}] E: [{}]".format(part.upper()[0], epoch))

        # switch to train mode
        model.eval()
        global_model.eval()
        routingNet.eval()
        if part=='gating':
            routingNet.train()
        elif part=='teacher':
            global_model.train()
        elif part=='student':
            model.train()
            if (args.base_method == 'hybrid_kd_inst') and (args.use_alt_min == 0):
                routingNet.train()

        end = time.time()

        for i, (images, target) in enumerate(train_loader):
            args.epoch_i = i
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            g_images = images #g_images.cuda(non_blocking=True)
            g_target = target #g_target.cuda(non_blocking=True)

            optimizer.zero_grad()
            s_optimizer.zero_grad()
            t_optimizer.zero_grad()

            loss, s_logits, t_logits, s_pred, t_pred, inst_wt_loss, wt_loss, _gt_wt, aux_loss, sc_tic, sic_tc, sc_tc, sic_tic = get_loss( logger, g_images, images, target, model, global_model, routingNet, criterion, args, part=part, batch_idx=i, epoch=epoch )
        
            # measure accuracy and record loss
            acc1, acc5 = obtain_accuracy(s_logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            inst_losses.update(inst_wt_loss.item(), images.size(0))
            wt_losses.update(wt_loss.item(), images.size(0))
            aux_losses.update(aux_loss.item(), images.size(0))
            gt_losses.update(_gt_wt.item(), images.size(0))

            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            acc1, acc5 = obtain_accuracy(t_logits, target, topk=(1, 5))
            Ttop1.update(acc1.item(), images.size(0))
            Ttop5.update(acc5.item(), images.size(0))

            if not torch.isfinite(loss):
                print('Got inf loss.')
                continue

            # compute gradient and do SGD step
            loss.backward()
            if part=='gating':
                optimizer.step()
            elif part=='student':
                s_optimizer.step()
                if (args.base_method == 'hybrid_kd_inst') and (args.use_alt_min == 0):
                    optimizer.step()
            elif part=='teacher':
                t_optimizer.step()
            else:
                assert(1==2)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % args.print_freq == 0) or (i == len(train_loader)-1):
                #torch.cuda.empty_cache()
                progress.display(i)

    return losses.avg

def validate(logger, val_loader, model, global_model, routingNet, criterion, args, ):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    inst_losses = AverageMeter('InstLoss', ':.4e')
    wt_losses = AverageMeter('WtLoss', ':.4e')
    aux_losses = AverageMeter('Aux', ':.4e')
    gt_losses = AverageMeter('Gt', ':.4e')

    STtop1 = AverageMeter('STAcc@1', ':6.2f')

    l_sc_tc = AverageMeter('ScTcAcc@1', ':6.2f')
    l_sc_tic = AverageMeter('ScTicAcc@1', ':6.2f')
    l_sic_tc = AverageMeter('SicTcAcc@1', ':6.2f')
    l_sic_tic = AverageMeter('SicTicAcc@1', ':6.2f')

    Ttop1 = AverageMeter('TAcc@1', ':6.2f')
    Ttop5 = AverageMeter('TAcc@5', ':6.2f')

    oracleAtCov = AverageMeter('OracleAtCov', ':6.2f')
    tAtCov = AverageMeter('tAtCov', ':6.2f')
    sAtCov = AverageMeter('sAtCov', ':6.2f')
    lcov = AverageMeter('lcov', ':6.2f')
    gateAcc = AverageMeter('gateAcc', ':6.2f')

    progress = ProgressMeter(
        logger,
        len(val_loader),
        [batch_time, losses, top1, top5, Ttop1, Ttop5, gt_losses, inst_losses, wt_losses, aux_losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    global_model.eval()
    routingNet.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            args.epoch_i = i
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            g_images = images #g_images.cuda(non_blocking=True)
            g_target = target #g_target.cuda(non_blocking=True)

            loss, s_logits, t_logits, s_pred, t_pred, inst_wt_loss, wt_loss, _gt_wt, aux_loss, sc_tic, sic_tc, sc_tc, sic_tic = get_loss( logger, g_images, images, target, model, global_model, routingNet, criterion, args, part='student', batch_idx=i, )

            # measure accuracy and record loss
            acc1, acc5 = obtain_accuracy(s_logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            inst_losses.update(inst_wt_loss.item(), images.size(0))
            wt_losses.update(wt_loss.item(), images.size(0))
            aux_losses.update(aux_loss.item(), images.size(0))
            gt_losses.update(_gt_wt.item(), images.size(0))

            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            acc1, acc5 = obtain_accuracy(t_logits, target, topk=(1, 5))
            Ttop1.update(acc1.item(), images.size(0))
            Ttop5.update(acc5.item(), images.size(0))

            acc1, acc5 = obtain_accuracy(s_logits, torch.argmax( t_logits, dim=1 ), topk=(1, 5))
            STtop1.update(acc1.item(), images.size(0))

            l_sc_tc.update(sc_tc.item(), images.size(0))
            l_sc_tic.update(sc_tic.item(), images.size(0))
            l_sic_tc.update(sic_tc.item(), images.size(0))
            l_sic_tic.update(sic_tic.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if i % args.print_freq == 0:
            if (i % args.print_freq == 0) or (i == len(val_loader)-1):
                #torch.cuda.empty_cache()
                progress.display(i)

        logger.log('base_flops='+ str(args.base_flops) + ' -- global_flops=' + str(args.global_flops))
        # TODO: this should also be done with the ProgressMeter
        logger.log(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}  --- TAcc@1 {Ttop1.avg:.3f}  TAcc@5 {Ttop5.avg:.3f} -- S-T Agreement {STtop1.avg:.3f} -- ScTc {l_sc_tc.avg:.3f} -- ScTic {l_sc_tic.avg:.3f} -- SicTc {l_sic_tc.avg:.3f} -- SicTic {l_sic_tic.avg:.3f} '
              .format(top1=top1, top5=top5, Ttop1=Ttop1, Ttop5=Ttop5, STtop1=STtop1, l_sc_tc=l_sc_tc, l_sic_tc=l_sic_tc, l_sc_tic=l_sc_tic, l_sic_tic=l_sic_tic,))

    return top1.avg


def get_trainable_params(model):
    var = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        var.append( param )
    return var

def get_optimizer( lr, optim_type, args, model, wd = None ):
    use_wd = args.wd
    if wd is not None:
      use_wd = wd
    model_vars = get_trainable_params(model)
    print('len models before freeze --', len(model_vars), ' -- ', optim_type, ' lr=', lr, ' -- wd=', use_wd, ' --mom=', args.momentum)
    if optim_type == 'sgd':
        optimizer = torch.optim.SGD(model_vars, lr, momentum=args.momentum, weight_decay=use_wd)
    elif optim_type == 'asgd':
        optimizer = torch.optim.ASGD(model_vars, lr, t0=0.)
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model_vars, lr, weight_decay=use_wd)
    else:
        optimizer = torch.optim.Adam(model_vars, lr, weight_decay=use_wd)
    return optimizer


def get_mlr(lr_scheduler):
     return lr_scheduler.optimizer.param_groups[0]['lr']

def main_train_eval_loop( logger, args, model, global_model, routingNet, 
      train_loader, val_loader, 
      base_model_name, global_model_name, resume_checkpoint=None,
      start_epoch=0, epochs=40, steps = 10):

    args.lmbda_climb = True
    args.lmbda_climb_val = args.lmbda_dual
    args.epoch_i=0

    logger.log('len models before freeze --' + str(len(list(model.parameters()))) )
    logger.log('len Global models before freeze --' + str(len(list(global_model.parameters()))) )
    args.logger = logger

    criterion = nn.CrossEntropyLoss().cuda()
    s_optimizer = get_optimizer( args.s_lr, args.base_opt_type, args, model, wd=args.wd  )
    t_optimizer = get_optimizer( args.t_lr, args.global_opt_type, args, global_model )
    optimizer = get_optimizer( args.g_lr, args.routing_opt_type, args, routingNet, wd=args.r_wd )

    T = 4
    args.T = T
    args.steps = args.epochs // T

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs // T)
    s_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(s_optimizer, args.epochs // T)
    t_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t_optimizer, args.epochs // T)

    init_optimizer = get_optimizer( args.g_lr, args.routing_opt_type, args, routingNet, wd=args.r_wd )
    init_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(init_optimizer, max(1, args.init_epochs))

    best_acc1 = validate(logger, val_loader, model, global_model, routingNet, criterion, args, )
    best_state_dict = copy.deepcopy( model.state_dict() )

    dual_variable_original_val = args.lmbda_dual
    grad_g_so_far=1000.
    abs_loss_diff_so_far = 0
    min_loss_so_far=1000000.

    # Initialize g(x)
    for epoch in range(0, args.init_epochs):
        train_part(logger, train_loader, model, global_model, routingNet, criterion, init_optimizer, epoch, args, s_optimizer, t_optimizer, part='gating')
        init_scheduler.step(epoch)
        logger.log('\t\t LR=' + str(get_mlr(scheduler)) + ' init-LR=' + str(get_mlr(init_scheduler)) + ' -- base-LR=' + str(get_mlr(s_scheduler)) + ' -- global-LR=' + str(get_mlr(t_scheduler)) + ' -- best base-acc=' + str(best_acc1) )

    old_s = model.state_dict()
    old_g = routingNet.state_dict()
    new_s = model.state_dict()
    new_g = routingNet.state_dict()

    dual_update_step_correction_term( old_s, old_g, new_s, new_g, s_optimizer, optimizer, logger, args, train_loader, routingNet, global_model, model, epoch=1 )

    for epoch in range(start_epoch, epochs):
        if epoch>0 and (epoch%steps == 0):
            min_loss_so_far=1000000.
            abs_loss_diff_so_far = 0
            args.lmbda_dual = dual_variable_original_val 
            args.lmbda_climb = True
            args.lmbda_climb_val = args.lmbda_dual

        if epoch>0 and epoch% args.budget_Ti == 0:
            args.budget_g_max = args.budget_g_max * args.budget_g_gamma 
            args.budget_g = args.budget_g_max 

        if epoch>5 and epoch%5==0:
            update_lmbda_climb_val( args )

        #if epoch>5:
        if epoch>1:
            #dual_update_step( logger, args, train_loader, routingNet, global_model, model, epoch=epoch )

            dual_update_step_correction_term( old_s, old_g, new_s, new_g, s_optimizer, optimizer, logger, args, train_loader, routingNet, global_model, model, epoch=epoch )

            updated_budget = primal_budget_update_step( logger, args, train_loader, routingNet, global_model, model, epoch=epoch, abs_loss_diff_so_far=abs_loss_diff_so_far )
            if updated_budget:
                min_loss_so_far=1000000.
                abs_loss_diff_so_far = 0

        for kt in range(args.kt):
            bg_loss = train_part(logger, train_loader, model, global_model, routingNet, criterion, optimizer, epoch, args, s_optimizer, t_optimizer, part='student')

        if args.use_alt_min == 1:
          for kt in range(args.kg):
            train_part(logger, train_loader, model, global_model, routingNet, criterion, optimizer, epoch, args, s_optimizer, t_optimizer, part='gating')
            scheduler.step(epoch)

        old_s, old_g = new_s, new_g
        new_s, new_g = model.state_dict(), routingNet.state_dict()

        if bg_loss > (min_loss_so_far - 0.01):
            abs_loss_diff_so_far += 1
        else:    
            min_loss_so_far = bg_loss
            abs_loss_diff_so_far = 0    

        acc1 = validate(logger, val_loader, model, global_model, routingNet, criterion, args, )

        scheduler.step(epoch)
        s_scheduler.step(epoch)
        t_scheduler.step(epoch)
        logger.log('\t\t LR=' + str(get_mlr(scheduler)) + ' init-LR=' + str(get_mlr(init_scheduler)) + ' -- base-LR=' + str(get_mlr(s_scheduler)) + ' -- global-LR=' + str(get_mlr(t_scheduler)) + ' -- best base-acc=' + str(best_acc1) )

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_state_dict = copy.deepcopy( model.state_dict() )

        args.logger = None
        save_checkpoint({
                'epoch': epoch + 1,
                'args' : copy.deepcopy(args),
                'base_state_dict': model.state_dict(),
                'global_state_dict': global_model.state_dict(),
                'routing_state_dict': routingNet.state_dict(),
                'best_acc1': best_acc1,

                'scheduler' : scheduler.state_dict(),
                'init_scheduler' : init_scheduler.state_dict(),
                's_scheduler' : s_scheduler.state_dict(),
                't_scheduler' : t_scheduler.state_dict(),

                'optimizer' : optimizer.state_dict(),
                'init_optimizer' : optimizer.state_dict(),
                't_optimizer' : t_optimizer.state_dict(),
                's_optimizer' : s_optimizer.state_dict(),
            }, is_best, prefix=get_model_prefix( args )) 
        args.logger = logger

    model.load_state_dict( best_state_dict )    

def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_model( network, xloader, criterion, batch_size ):
    losses, top1, top5 = ( AverageMeter(), AverageMeter(), AverageMeter(),)
    network.eval()

    for i, (inputs, targets) in enumerate(xloader):

        inputs = inputs.cuda()
        targets = targets.cuda(non_blocking=True)
        features, logits, _ = network(inputs)

        loss = criterion(logits, targets)
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    return losses.avg, top1.avg, top5.avg

