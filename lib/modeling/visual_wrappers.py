import torch
from torch import nn
import numpy as np
from torch.nn.init import normal_, constant_

from . import backbone
from lib.ops import get_agg, build_aug_block

class VisualModelWrapper(nn.Module):
    def __init__(self, clip_length,
                 modality,
                 backbone_name='resnet101',
                 backbone_type='2D',
                 new_length=None,
                 agg_fun='avg',
                 before_softmax=True,
                 dropout=0.8,
                 crop_num=1, 
                 partial_bn=True, 
                 fc_sche=False, 
                 custom_flag=False, 
                 pretrained=False,
                 module_name_list=None,
                 pretrain_path=None,
                 aug_flag=False):
        super(VisualModelWrapper, self).__init__()
        self.modality = modality
        self.backbone_name = backbone_name
        self.clip_length = clip_length
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.custom_flag = custom_flag
        self.agg_fun = agg_fun
        self.backbone_type = backbone_type
        self.module_name_list = module_name_list
        self.pretrained = pretrained
        self.fc_sche = fc_sche
        self.pretrain_path = pretrain_path
        self.aug_flag = aug_flag
        if not before_softmax and agg_fun != 'avg':
            raise ValueError("Only avg aggregattion can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length


        # -------------------------------------------------------------------------
        # prepare backbone
        # -------------------------------------------------------------------------
        self._prepare_base_model(backbone_name)

        # -------------------------------------------------------------------------
        # prepare video model
        # -------------------------------------------------------------------------
        self.feature_dim = self._prepare_video_model()  # 1024 dimension

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")

        # -------------------------------------------------------------------------
        # prepare reasonling module
        # -------------------------------------------------------------------------
        self.aggregation = get_agg(agg_fun=self.agg_fun, model_type=self.backbone_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn

    def forward(self, x):
        '''
        print('=================================')
        print('Shape of X: ', x.shape)
        print('=================================')
        dataset input shape (B, TxC, H, W)
        2d (BxT, C, H, W)
        3d (B, C, T, H, W)
        '''
        sample_cha = (3 if self.modality == "RGB" else 2) * self.new_length
        
        if self.backbone_type == '2D':
            x = x.view((-1, sample_cha) + x.shape[-2:])
    
        base_out = self.base_model(x)
        base_out = base_out.view(-1, self.feature_dim)        

        if self.backbone_type == '2D':
            base_out = base_out.view((-1, self.clip_length) + base_out.size()[1:])
            return self.aggregation(base_out).squeeze(1)
        elif self.backbone_type == '3D':
            return base_out
        else:
            raise ValueError("Backbone Type Error!")
        

    def _prepare_video_model(self):
        if self.backbone_name == 'S3D':
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[0].in_channels
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, Identity())
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            

        if self.aug_flag:
            if self.module_name_list == None:
                if self.backbone_name == 'bninception':
                    self.module_name_list = ['inception3b', 'inception4c', 'inception5b']
                elif self.backbone_name == 'inception_v3':
                    self.module_name_list = ['Mixed_5b', 'Mixed_6b', 'Mixed_7c']
                elif 'resnet' in self.backbone_name:
                    self.module_name_list = ['layer2', 'layer3', 'layer4']
                elif self.backbone_name == 'S3D':
                    self.module_name_list = ['base.5', 'base.9', 'base.14']
            self.base_model = build_aug_block(self.base_model, self.module_name_list,
                    n_segments=self.clip_length)

        return feature_dim

    def _prepare_base_model(self, backbone_name):
        
        if self.backbone_type == '2D':
            self.base_model = getattr(backbone.backbone_2d, backbone_name)()
        elif self.backbone_type == '3D':
            self.base_model = getattr(backbone.backbone_3d, backbone_name)()
        else:
            raise ValueError("Only support 2D or 3D backbone")

        if self.pretrained and self.pretrain_path != 'none':
            print(self.pretrained, self.pretrain_path)
            self.base_model.load_state_dict(torch.load(self.pretrain_path))

        if backbone_name == ' InceptionV3':
            self.base_model.last_layer_name = 'classif'
        else:
            self.base_model.last_layer_name = 'fc'

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        fc_weight = []
        fc_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_sche:
                    fc_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_sche:
                        fc_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': fc_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "fc_weight"},
            {'params': fc_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "fc_bias"}, 
        ]


    def _construct_flow_model(self, base_model):

        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data 
        layer_name = list(container.state_dict().keys())[0][:-7] 

        setattr(container, layer_name, new_conv)
        return base_model

    def _inflate_module(self):
        for name, module in self.base_model.named_children():
            if isinstance(module, nn.Conv2d):
                kernel_size = (module.shape[1], 2*self.new_length, module.shape[2], module.shape[3])

                new_kernel_weight = module.weight.clone().data.mean(dim=1, keepdim=True).expand(kernel_size).contiguous()
                new_conv = nn.Conv2d(2 * self.new_length, module.out_channels,
                            module.kernel_size, module.stride, module.padding,
                            bias=True if len(params) == 2 else False)
                new_conv.weight.data = new_kernel_weight
                if module.bias:
                    new_conv.bias.data = module.bias.data
                setattr(self.base_model, name, new_conv)
                break
        return 0


    def __repr__(self):
        fmt_str = 'Initializing  video model with base backbone: {}.\n'.format(self.backbone_name)
        fmt_str += 'video model Configurations:\n'
        fmt_str += '    input_modality:     {}\n'.format(self.modality)
        fmt_str += '    clip length:       {}\n'.format(self.clip_length)
        fmt_str += '    aggregation_function:   {}\n'.format(self.agg_fun)
        fmt_str += '    dropout_ratio:      {}'.format(self.dropout)
        return fmt_str

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
