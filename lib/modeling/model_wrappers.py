import torch
from torch import nn
import numpy as np
from torch.nn.init import normal_, constant_

from . import backbone
from lib.ops import get_agg  

class VideoModelWrapper(nn.Module):
    def __init__(self, num_class,
                 clip_length,
                 modality,
                 backbone_name='resnet101',
                 backbone_type='2D',
                 new_length=None,
                 agg_fun='avg',
                 before_softmax=True,
                 dropout=0.8,
                 crop_num=1, 
                 partial_bn=True, fc_sche=False, 
                 reason_flag=False, module_name_list=None,
                 pretrained=False, pretrain_path=None):
        super(VideoModelWrapper, self).__init__()
        self.modality = modality
        self.backbone_name = backbone_name
        self.clip_length = clip_length
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.reason_flag = reason_flag
        self.module_name_list = module_name_list
        self.agg_fun = agg_fun
        self.backbone_type = backbone_type
        self.pretrained = pretrained
        self.fc_sche = fc_sche
        self.pretrain_path = pretrain_path
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
        feature_dim = self._prepare_video_model(num_class)  # 1024 dimension

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
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        if self.backbone_type == '2D':
            x = x.view((-1, sample_len) + x.shape[-2:])
    
        base_out = self.base_model(x)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.backbone_type == '2D':
            base_out = base_out.view((-1, self.clip_length) + base_out.size()[1:])
            return self.aggregation(base_out).squeeze(1)
        elif self.backbone_type == '3D':
            return base_out
        else:
            raise ValueError("Backbone Type Error!")
        

    def _prepare_video_model(self, num_class):
        if self.backbone_name == 'S3D':
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[0].in_channels
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, backbone_name):
        
        if self.backbone_type == '2D':
            self.base_model = getattr(backbone.backbone_2d, backbone_name)()
        elif self.backbone_type == '3D':
            self.base_model = getattr(backbone.backbone_3d, backbone_name)()
        else:
            raise ValueError("Only support 2D or 3D backbone")

        if self.pretrained and self.pretrain_path is not None:
            self.base_model.load_state_dict(torch.load(self.pretrain_path))
        self.base_model.last_layer_name = 'fc'


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(VideoModelWrapper, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D or BatchNorm3D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


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
