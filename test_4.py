import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
Image.MAX_IMAGE_PIXELS = None



decoder_0 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),  
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 32, (3, 3)),
    )

decoder_1 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(1024, 256, (3, 3)),  
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 32, (3, 3)),
    )

convs = nn.Sequential(
    nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 32, (3, 3)), 
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(32, 16, (3, 3)), 
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(16, 3, (3, 3)),
    nn.ReLU()
    )


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


re = nn.Sequential(
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
)





class Net(nn.Module):
    def __init__(self, encoder, decoder_0, decoder_1, convs):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder_0 = decoder_0
        self.decoder_1 = decoder_1
        self.convs = convs
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        # print(input.size(), target.size())
        target = target.unsqueeze(dim = 0)
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std_0(input)
        target_mean, target_std = calc_mean_std_0(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, cm, style, sm, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        style_feat = style_feats[-1]
        cm_re = re(cm)
        sm_re = re(sm)
        content_mask = cm_re
        style_mask = sm_re
        
        loss_c = 0.0
        loss_s = 0.0
        g_t = []
        for i in range(8):
            
            t_g = adaptive_instance_normalization_0(content_feat[i,:,:,:].unsqueeze(0), style_feat[i,:,:,:].unsqueeze(0))
            g_t_i_0 = self.decoder_0(t_g)
            
            if style_mask[i,:,:,:].view(1,1,-1).sum(dim=2) != 0:
                one = torch.ones(1,512,32,32).cuda()
                content_mask_temp = content_mask[i,:,:,:].unsqueeze(0)
                content_mask_temp = content_mask_temp.repeat(1,512,1,1)
                style_mask_temp = style_mask[i,:,:,:].unsqueeze(0)
                style_mask_temp = style_mask_temp.repeat(1,512,1,1)
                
                t_0 = adaptive_instance_normalization(content_feat[i,:,:,:].unsqueeze(0)*content_mask_temp, content_mask_temp, 
                    style_feat[i,:,:,:].unsqueeze(0)*style_mask_temp, style_mask_temp)
                t_1 = adaptive_instance_normalization(content_feat[i,:,:,:].unsqueeze(0)-content_feat[i,:,:,:].unsqueeze(0)*content_mask_temp, 
                    one-content_mask_temp, style_feat[i,:,:,:].unsqueeze(0)-style_feat[i,:,:,:].unsqueeze(0)*style_mask_temp,
                    one-style_mask_temp)
                
                t_l  = torch.cat((t_0, t_1), dim=1)
                # t = alpha * t + (1 - alpha) * content_feat 
                g_t_i_1 = self.decoder_1(t_l)  
            else:
                g_t_i_1 = g_t_i_0

            g_t_i = torch.cat((g_t_i_0, g_t_i_1), dim=1)
            g_t_i = self.convs(g_t_i)#genreated image 三通道图像
            g_t.append(g_t_i) 
            g_t_feats = self.encode_with_intermediate(g_t_i)

            loss_c += self.calc_content_loss(g_t_feats[-1], content_feat[i,:,:,:])# 0810
            loss_s += self.calc_style_loss(g_t_feats[0], style_feats[0][i,:,:,:].unsqueeze(0))
            for q in range(1, 4):
                loss_s += self.calc_style_loss(g_t_feats[q], style_feats[q][i,:,:,:].unsqueeze(0))
        
        loss_c = loss_c / 8
        loss_s = loss_s / 8
        
        return g_t, loss_c, loss_s

def calc_mean_std_0(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean_std(feat, mask, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    size = feat.view(N, C, -1).size()
    mask_sum = mask.view(N, C, -1).sum(dim=2) + 1e-10
    feat_sum = feat.view(N, C, -1).sum(dim=2) 
    feat_mean = (feat_sum / mask_sum).view(N, C, 1, 1)
    
    feat_var_0 = feat.view(N, C, -1) - feat_mean.view(N, C, 1).expand(size)
    #print(feat_var_0.size())
    feat_var_1 = torch.pow(feat_var_0, 2)
    #print(feat_var_1.size())
    feat_var = torch.bmm(feat_var_1.view(N*C, 1, -1), mask.view(N*C, -1, 1)).sum(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)

    return feat_mean, feat_std

def adaptive_instance_normalization_0(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_0(style_feat)
    content_mean, content_std = calc_mean_std_0(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adaptive_instance_normalization(content_feat, content_mask, style_feat, style_mask):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat, style_mask)
    content_mean, content_std = calc_mean_std(content_feat, content_mask)

    normalized_feat = ((content_feat - content_mean.expand(
        size)) / content_std.expand(size)) * content_mask

    return (normalized_feat * style_std.expand(size) + style_mean.expand(size)) * content_mask


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())



def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize([size,size]))
    #if crop:
        #transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


def style_transfer(vgg, decoder_0, decoder_1, convs, content, cm, style, sm, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    style_feat = vgg(style)
    content_feat = vgg(content)
    
    cm_re = re(cm)
    sm_re = re(sm)
    content_mask = cm_re
    style_mask = sm_re
    
    t_g = adaptive_instance_normalization_0(content_feat, style_feat)
    g_t_0 = decoder_0(t_g)
    
    if style_mask.view(1,1,-1).sum(dim=2) != 0:
        one = torch.ones(1,512,64,64).cuda()
        content_mask_temp = content_mask.repeat(1,512,1,1)
        style_mask_temp = style_mask.repeat(1,512,1,1)
        
        t_0 = adaptive_instance_normalization(content_feat*content_mask_temp, content_mask_temp, 
            style_feat*style_mask_temp, style_mask_temp)
        t_1 = adaptive_instance_normalization(content_feat-content_feat*content_mask_temp, 
            one-content_mask_temp, style_feat-style_feat*style_mask_temp, one-style_mask_temp)
        
        t_l  = torch.cat((t_0, t_1), dim=1)
        # t = alpha * t + (1 - alpha) * content_feat 
        g_t_1 = decoder_1(t_l)  
    else:
        g_t_1 = g_t_0

    g_t = torch.cat((g_t_0, g_t_1), dim=1)
    g_t = convs(g_t)#genreated image 三通道图像
    
    return g_t
   
    # content_f = vgg(content)
    # style_f = vgg(style)
    # if interpolation_weights:
    #     _, C, H, W = content_f.size()
    #     feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
    #     base_feat = adaptive_instance_normalization(content_f, style_f)
    #     for i, w in enumerate(interpolation_weights):
    #         feat = feat + w * base_feat[i:i + 1]
    #     content_f = content_f[0:1]
    # else:
    #     feat = adaptive_instance_normalization(content_f, style_f)
    # feat = feat * alpha + content_f * (1 - alpha)
    # return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,default='',
                    help='File path to the content image')
parser.add_argument('--style', type=str,default='',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--content_dir', type=str, default='/student_1/lyt/measurement/testC',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/student_1/lyt/measurement/testS',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='/student_1/lyt/measurement/Ours/vgg_normalised.pth')
parser.add_argument('--content_mask_dir', type=str, default='/student_1/lyt/measurement/saliency_mask/test_res_07-24-16-08-53/testC/mask',
                    help='Directory path to a batch of content_mask images')
parser.add_argument('--style_mask_dir', type=str, default='/student_1/lyt/measurement/saliency_mask/test_res_07-24-16-13-13/testS/mask',
                    help='Directory path to a batch of style_mask images')
parser.add_argument('--decoder_0', type=str, default='/student_1/lyt/measurement/Ours/experiments_11/decoder_0_iter_160000.pth.tar')
parser.add_argument('--decoder_1', type=str, default='/student_1/lyt/measurement/Ours/experiments_11/decoder_1_iter_160000.pth.tar')
parser.add_argument('--convs', type=str, default='/student_1/lyt/measurement/Ours/experiments_11/convs_iter_160000.pth.tar')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='./output_11',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content!='':
    content_paths = [Path(args.content)]
else:
    # content_dir = Path(args.content_dir)
    # content_paths = [f for f in content_dir.glob('*')]
    content_paths = os.listdir(args.content_dir)

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style!='':
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    # style_dir = Path(args.style_dir)
    # style_paths = [f for f in style_dir.glob('*')]
    style_paths = os.listdir(args.style_dir)

decoder_0 = decoder_0
decoder_1 = decoder_1
convs = convs
vgg = vgg

decoder_0.eval()
decoder_1.eval()
convs.eval()
vgg.eval()

decoder_0.load_state_dict(torch.load(args.decoder_0))
decoder_1.load_state_dict(torch.load(args.decoder_1))
convs.load_state_dict(torch.load(args.convs))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder_0.to(device)
decoder_1.to(device)
convs.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)
 
i=1

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        for style_path in style_paths:
            content_image = content_tf(Image.open(os.path.join(args.content_dir, content_path)).convert('RGB'))
            style_image = style_tf(Image.open(os.path.join(args.style_dir, style_path)).convert('RGB'))
            content_mask = content_tf(Image.open(os.path.join(args.content_mask_dir, content_path.split('.')[0]+'.png')).convert('1'))
            style_mask = style_tf(Image.open(os.path.join(args.style_mask_dir, style_path.split('.')[0]+'.png')).convert('1'))
            if (style_image.shape[0]==1):
                style_image = style_image.repeat(3,1,1)
            if (style_image.shape[0]>3):
                style_image = style_image[0:3,:,:]
            if(not style_image.shape[0]==3):
                print("维度异常:",style_image.shape,content_path)

            if args.preserve_color:
                style_image = coral(style_image, content_image)

            style_image = style_image.to(device).unsqueeze(0)
            content_image = content_image.to(device).unsqueeze(0)
            style_mask = style_mask.to(device).unsqueeze(0)
            content_mask = content_mask.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder_0, decoder_1, convs, content_image, content_mask, style_image, style_mask,
                                        args.alpha)
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.split('.')[0], style_path.split('.')[0], args.save_ext)
            save_image(output, str(output_name))
            print(i)
            i=i+1