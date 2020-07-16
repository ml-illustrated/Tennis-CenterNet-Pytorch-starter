import torch
from torch import nn
from types import MethodType
# from .utils import load_state_dict_from_url, BN_MOMENTUM

# from nets.layers import TemporalShift, TemporalShiftV2
#from nets.layers import TemporalShift, TemporalShiftRight, TemporalShiftSingleBatch, TemporalShiftV2, TemporalShiftV2SingleBatch, AddBatch, RemoveBatch, ConvUpBlock, TemporalShiftRightSingleBatch, TemporalShiftRight2, TemporalShiftRight2SingleBatch

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
        
BN_MOMENTUM = 0.1


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.identity = self.use_res_connect
        self.has_point_wise = (expand_ratio != 1)

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    '''
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
    '''
    def __init__(
            self,
            num_classes,
            head_conv=24,
            width_mult=1.0,
            inverted_residual_setting=None,
            round_nearest=8,
            block=None,
            tsm_mode=None,
            export_mode=False,
            deconv_size=None, # defaults to same width as head_conv
    ):

        self.deconv_with_bias = False
        self.tsm_mode = tsm_mode
        self.export_mode=export_mode
    
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        # last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        # self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        # features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.inplanes = input_channel

        if self.tsm_mode:
            new_forward = None
            if export_mode:
                if tsm_mode == 'v2':
                    block = TemporalShiftV2SingleBatch
                elif tsm_mode == 'right':
                    block = TemporalShiftRightSingleBatch
                    def new_forward( self, x, prev_shift_out ):
                        x, slice_shift_out = self.shift_layer( x, prev_shift_out )
                        return x + self.conv(x), slice_shift_out
                elif tsm_mode == 'right2':
                    block = TemporalShiftRight2SingleBatch
                    def new_forward( self, x, prev_shift_out_x2 ):
                        x, slice_shift_out_x2 = self.shift_layer( x, prev_shift_out_x2 )
                        return x + self.conv(x), slice_shift_out_x2
                elif tsm_mode == 'orig':
                    block = TemporalShiftSingleBatch
                else:
                    assert False, 'Unknown TSM mode!'
            else:
                if tsm_mode == 'v2':
                    block = TemporalShiftV2
                elif tsm_mode == 'right':
                    block = TemporalShiftRight
                elif tsm_mode == 'right2':
                    block = TemporalShiftRight2
                elif tsm_mode == 'orig':
                    block = TemporalShift
                else:
                    assert False, 'Unknown TSM mode!'
                    
            if new_forward == None:
                def new_forward( self, x):
                    # print( 'new_forward: ', x.shape )
                    x = self.shift_layer( x )
                    return x + self.conv(x)
                    
            layer_idx=0
            for m in self.features:
                if isinstance(m, InvertedResidual) and m.has_point_wise and m.identity:
                    print('Adding temporal shift... idx {} conv[0] {}'.format(layer_idx, m.conv[0][0].in_channels ))
                    fold = m.conv[0][0].in_channels//8
                    m.shift_layer = block( fold=fold )
                    m.forward = MethodType( new_forward, m )
                    '''
                    if block in ( TemporalShiftRightSingleBatch, TemporalShiftRight2SingleBatch ):
                        m.shift_layer = block( fold=fold )
                        m.forward = MethodType( new_forward, m )
                    else:
                        m.conv[0] = block(m.conv[0], fold=fold)
                    '''
                layer_idx+=1

        deconv_layer_channels = deconv_size if deconv_size else head_conv # default to use same size as head_conv
        self.deconv_layers = self._make_deconv_layer(
            3,
            [deconv_layer_channels*4, deconv_layer_channels*2, deconv_layer_channels],
            [4, 4, 4],
        )
        # self.final_layer = []

        if head_conv > 0:
            if True:
                # heatmap layers
                self.hmap = nn.Sequential(nn.Conv2d(deconv_layer_channels, head_conv, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(head_conv, num_classes, kernel_size=1))
                self.hmap[-1].bias.data.fill_(-2.19)
                # regression layers
                self.regs = nn.Sequential(nn.Conv2d(deconv_layer_channels, head_conv, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(head_conv, 2, kernel_size=1))
                self.w_h_ = nn.Sequential(nn.Conv2d(deconv_layer_channels, head_conv, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(head_conv, 2, kernel_size=1))
            
            # self.hmap[2].name = 'output_hmap'
            self.regs[2].name = 'output_regs'
            self.w_h_[2].name = 'output_w_h_'
        else:
            # heatmap layers
            self.hmap = nn.Conv2d(in_channels=deconv_layer_channels, out_channels=num_classes, kernel_size=1)
            # regression layers
            self.regs = nn.Conv2d(in_channels=deconv_layer_channels, out_channels=2, kernel_size=1)
            self.w_h_ = nn.Conv2d(in_channels=deconv_layer_channels, out_channels=2, kernel_size=1)
        
        '''
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        '''

    def init_weights(self, arch, pretrained=True):
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.hmap.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.constant_(m.bias, -2.19)
        for m in self.regs.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.w_h_.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            model_url = model_urls[arch]
            pretrained_state_dict = load_state_dict_from_url(model_url, progress=True)
            self.load_state_dict(pretrained_state_dict, strict=False)
                    
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        layer__up_size = [ (24, 24), (48, 48), (96, 96) ]
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            # print( 'deconv ', i, num_kernels[i], num_filters[i], kernel, padding, output_padding ) #  deconv  0 4 64 4 1 0

            planes = num_filters[i]
            if False: # REVERT TO ORIGINAL ConvTranspose2d TESTING ConvUpBlock
                # layers.append( ConvUpBlock( self.inplanes, planes ) )
                layers.append( ConvUpBlock( self.inplanes, planes, layer__up_size[i], export_mode=self.export_mode ) )
            else:
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=self.inplanes,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=self.deconv_with_bias))
                layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
                layers.append(nn.ReLU(inplace=True))
                                         
            self.inplanes = planes

        return nn.Sequential(*layers)
                
    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # features:  torch.Size([16, 320, 13, 13]) # w/ 16 images input

        # print( 'add_batch: ', x.shape )
        # add_batch:  torch.Size([2, 320, 8, 13, 13]) # w/ 16 images input
        
        x = self.deconv_layers(x)
        # print( 'deconv: ', x.shape )
        # deconv:  torch.Size([2, 24, 8, 104, 104]) # w/ 16 images input

        out = [self.hmap(x), self.regs(x), self.w_h_(x)]

        # print( 'OUT: ', out[0].shape, out[1].shape, out[2].shape ) 
        # w/ 2 batches: torch.Size([2, 3, 8, 104, 104]) torch.Size([2, 2, 8, 104, 104]) torch.Size([2, 2, 8, 104, 104])

        # !! REVERT to ORIGINAL return out
        return [out]
        
    def forward(self, x):
        return self._forward_impl(x)


    

'''
def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    model = MobileNetV2(**kwargs)
    return model
'''

def get_mobilenetv2(num_classes,width_mult=1.0, head_conv=24, pretrained=True, export_mode=False, deconv_size=None):
    arch = 'mobilenet_v2'
    # model = mobilenet_v2( heads=heads, head_conv=head_conv)
    model = MobileNetV2( num_classes=num_classes, width_mult=width_mult, head_conv=head_conv, export_mode=export_mode, deconv_size=deconv_size )
    model.init_weights(arch, pretrained=pretrained)
    return model


def get_mobilenetv2_tsm(num_classes,width_mult=1.0, head_conv=24, pretrained=True, export_mode=False, deconv_size=None ):
    arch = 'mobilenet_v2'
    # model = mobilenet_v2( heads=heads, head_conv=head_conv)
    model = MobileNetV2( num_classes=num_classes, width_mult=width_mult, head_conv=head_conv, tsm_mode='orig', export_mode=export_mode, deconv_size=deconv_size )
    model.init_weights(arch, pretrained=pretrained)
    return model


def get_mobilenetv2_tsmv2(num_classes,width_mult=1.0, head_conv=24, pretrained=True, export_mode=False, deconv_size=None):
    arch = 'mobilenet_v2'
    model = MobileNetV2( num_classes=num_classes, width_mult=width_mult, head_conv=head_conv, tsm_mode='v2', export_mode=export_mode )
    model.init_weights(arch, pretrained=pretrained)
    return model

def get_mobilenetv2_tsmright(num_classes,width_mult=1.0, head_conv=24, pretrained=True, export_mode=False, deconv_size=None):
    arch = 'mobilenet_v2'
    model = MobileNetV2( num_classes=num_classes, width_mult=width_mult, head_conv=head_conv, tsm_mode='right', export_mode=export_mode, deconv_size=deconv_size )
    model.init_weights(arch, pretrained=pretrained)
    return model

def get_mobilenetv2_tsmright2(num_classes,width_mult=1.0, head_conv=24, pretrained=True, export_mode=False, deconv_size=None):
    arch = 'mobilenet_v2'
    model = MobileNetV2( num_classes=num_classes, width_mult=width_mult, head_conv=head_conv, tsm_mode='right2', export_mode=export_mode, deconv_size=deconv_size )
    model.init_weights(arch, pretrained=pretrained)
    return model


def get_mobilenetv2_tsmright_for_export(num_classes, *args, **kwargs):
    return get_mobilenetv2_tsm_with_shift_for_export(tsm_mode='right', num_classes=num_classes, *args, **kwargs )

def get_mobilenetv2_tsmright2_for_export(num_classes, *args, **kwargs):
    return get_mobilenetv2_tsm_with_shift_for_export(tsm_mode='right2', num_classes=num_classes, *args, **kwargs )


def get_mobilenetv2_tsm_with_shift_for_export( tsm_mode, num_classes, width_mult=1.0, head_conv=24, pretrained=True, export_mode=True, deconv_size=None):
    
    from types import MethodType
    arch = 'mobilenet_v2'
    
    model = MobileNetV2( num_classes=num_classes, width_mult=width_mult, head_conv=head_conv, tsm_mode=tsm_mode, export_mode=export_mode, deconv_size=deconv_size )

    shift_width = 2 if tsm_mode=='right2' else 1
    shift_buffers = [torch.zeros([1, shift_width*3, 96, 96]),
		    torch.zeros([1, shift_width*4, 48, 48]),
                    torch.zeros([1, shift_width*4, 48, 48]),
                    torch.zeros([1, shift_width*8, 24, 24]),
                    torch.zeros([1, shift_width*8, 24, 24]),
                    torch.zeros([1, shift_width*8, 24, 24]),
                    torch.zeros([1, shift_width*12, 24, 24]),
                    torch.zeros([1, shift_width*12, 24, 24]),
	            torch.zeros([1, shift_width*20, 12, 12]),
                    torch.zeros([1, shift_width*20, 12, 12])]
    
    def new_forward(self, x, *shift_buffers):
        
        shift_buffer_idx = 0
        out_buffer = []
        for f in self.features:
            if hasattr(f, 'shift_layer'): # isinstance(f, TemporalShiftRightSingleBatch):
                x, shift_out = f(x, shift_buffers[shift_buffer_idx])
                shift_buffer_idx += 1
                out_buffer.append(shift_out)
            else:
                x = f(x)
        
        x = self.deconv_layers(x)

        out = [self.hmap(x), self.regs(x), self.w_h_(x)] + out_buffer
        
        return out

    model.forward = MethodType(new_forward, model)
    
    return model, shift_buffers


if __name__ == '__main__':
    device='cuda'
    device='cpu'

    # width_mult=1.0
    # head_conv = 24
    num_classes = 3
    input_shape = (16,3,384,384)

    # model = get_mobilenetv2_tsm( num_classes )
    # model = get_mobilenetv2( num_classes )
    # model = get_mobilenetv2_tsmright( num_classes )
    # model, shift_buffers = get_mobilenetv2_tsmright_for_export(num_classes)
    model, shift_buffers = get_mobilenetv2_tsmright2_for_export(num_classes)
    input_shape = (1,3,384,384)
    # print( model )

    model = model.to(device)
    # from torchsummary import summary
    # summary( model, input_shape )

    input = torch.zeros( input_shape )
    outputs = model(input.to(device), *shift_buffers)
    for o in outputs:
        print( 'output: ', o.shape )

