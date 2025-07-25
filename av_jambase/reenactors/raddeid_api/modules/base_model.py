"""This script defines the base network model for Deep3DFaceRecon_pytorch
"""

import os
import numpy as np
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.parallel_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass
    @abstractmethod
    def read_data_batch(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)


        # self.print_networks(opt.verbose)

    def parallelize(self, convert_sync_batchnorm=True):
        if not self.opt.use_ddp:
            for name in self.parallel_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))
        else:
            for name in self.model_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    if convert_sync_batchnorm:
                        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
                    setattr(self, name, torch.nn.parallel.DistributedDataParallel(module.to(self.device),
                        device_ids=[self.device.index],
                        find_unused_parameters=True, broadcast_buffers=True))

            # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
            for name in self.parallel_names:
                if isinstance(name, str) and name not in self.model_names:
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))

        # put state_dict of optimizer to gpu device
        if self.opt.phase != 'test':
            if self.opt.continue_train:
                for optim in self.optimizers:
                    for state in optim.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)

    def data_dependent_initialize(self, data):
        pass

    def train(self):
        """Make models train mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        """Make models eval mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test_recon(self,net_recog):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        """Make models eval mode"""

        for name in self.model_names:
           # print(name)
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()
        with torch.no_grad():
            self.forward(epoch=161)

        compute=False
        if compute:
            id_sim,ssim,lpips,psnr,fsim,l2_dist, id_enc,ssim_enc,lpips_enc,psnr_enc,fsim_enc,l2_dist_enc=self.compute_visuals_test(net_recog)
            return id_sim,ssim,lpips,psnr,fsim,l2_dist,id_enc,ssim_enc,lpips_enc,psnr_enc,fsim_enc,l2_dist_enc
        else:
            flag=self.compute_visuals_test(net_recog)
            return flag

    #===================================================
    def test_recon_frame(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        """Make models eval mode"""

        for name in self.model_names:
           # print(name)
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()
        with torch.no_grad():
            #self.forward(epoch=99)
            shape_first_frm=self.compute_shape_first_frm()
        return shape_first_frm
    #-------------------------------------------------
    def test_recon_video(self,net_recog,tst_coeffs,CPWD1,CPWD2, global_pose=None, color_override=False):
        for name in self.model_names:
           # print(name)
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()
        with torch.no_grad():
            #self.forward(epoch=99)
            #output_vis_deid,output_vis_ori,pred_lm_enc1,pred_lm,deid_sim
           recon_image_deid,recon_image_ori,rec_lm_deid,pred_lm_ori,deid_sim,ori_sim,pred_mask,pred_mask_enc1= self.compute_visuals_test_video(net_recog,tst_coeffs,CPWD1,CPWD2,
                                                                                                         global_pose=global_pose, color_override=color_override)
        return recon_image_deid,recon_image_ori,rec_lm_deid,pred_lm_ori,deid_sim,ori_sim,pred_mask,pred_mask_enc1
    #===================================================
    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward(epoch=99)
            self.compute_visuals(epoch=99)

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def compute_visuals_test(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def compute_shape_first_frm(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self, name='A'):
        """ Return image paths that are used to load current data"""
        return self.image_paths if name =='A' else self.image_paths_B

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']


        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()

        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)[:, :3, ...]
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        save_filename = 'epoch_%s.pth' % (epoch)
        save_path = os.path.join(self.save_dir, save_filename)

        save_dict = {}
        for name in self.model_names:
            #print(name)
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel) or isinstance(net,
                        torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                save_dict[name] = net.state_dict()


        for i, optim in enumerate(self.optimizers):
            save_dict['opt_%02d'%i] = optim.state_dict()

        for i, sched in enumerate(self.schedulers):
            save_dict['sched_%02d'%i] = sched.state_dict()

        torch.save(save_dict, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if self.opt.isTrain and self.opt.pretrained_name is not None:
            load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
        else:
            load_dir = self.save_dir
        load_filename = 'epoch_%s.pth' % (epoch)
        load_path = os.path.join(load_dir, load_filename)
        state_dict = torch.load(load_path, map_location=self.device)
        print('loading the model from %s' % load_path)

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name])

        if self.opt.phase != 'test':
            if self.opt.continue_train:
                print('loading the optim from %s' % load_path)
                for i, optim in enumerate(self.optimizers):

                    optim.load_state_dict(state_dict['opt_%02d'%i])
                    # if i>0:

                    #    optim.param_groups[0]['initial_lr']=0.00001
                    #    optim.param_groups[0]['lr']=0.00001
                    #    #optim.param_groups[0]['amsgrad']=True

                    #    optim.param_groups[0]['betas']= (0.9, 0.999)

                    #    print('sucess update',optim)




                try:
                    print('loading the sched from %s' % load_path)
                    for i, sched in enumerate(self.schedulers):
                        sched.load_state_dict(state_dict['sched_%02d'%i])
                except:
                    print('Failed to load schedulers, set schedulers according to epoch count manually')
                    for i, sched in enumerate(self.schedulers):
                        sched.last_epoch = self.opt.epoch_count - 1




    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}
