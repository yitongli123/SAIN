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

def test_transform(size):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize([size,size]))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder_0, decoder_1, convs, content, cm, style, sm):
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
        g_t_1 = decoder_1(t_l)  
    else:
        g_t_1 = g_t_0

    g_t = torch.cat((g_t_0, g_t_1), dim=1)
    g_t = convs(g_t)
    
    return g_t


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='./image/testC',
                    help='Directory path to content images')
parser.add_argument('--style_dir', type=str, default='./image/testS',
                    help='Directory path to style images')
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')
parser.add_argument('--content_mask_dir', type=str, default='./mask/testC',
                    help='Directory path to content masks')
parser.add_argument('--style_mask_dir', type=str, default='./mask/testS',
                    help='Directory path to style masks')
parser.add_argument('--decoder_0', type=str, default='./experiments/decoder_0_iter_160000.pth.tar')
parser.add_argument('--decoder_1', type=str, default='./experiments/decoder_1_iter_160000.pth.tar')
parser.add_argument('--convs', type=str, default='./experiments/convs_iter_160000.pth.tar')
# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='./output',
                    help='Directory to save the output images')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

assert (args.content_dir)
content_paths = os.listdir(args.content_dir)

assert (args.style_dir)
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

content_tf = test_transform(args.content_size)
style_tf = test_transform(args.style_size)
 
i=1
for content_path in content_paths:
    for style_path in style_paths:
        content_image = content_tf(Image.open(os.path.join(args.content_dir, content_path)).convert('RGB'))
        style_image = style_tf(Image.open(os.path.join(args.style_dir, style_path)).convert('RGB'))
        content_mask = content_tf(Image.open(os.path.join(args.content_mask_dir, content_path.split('.')[0]+'.jpg')).convert('1')) 
        style_mask = style_tf(Image.open(os.path.join(args.style_mask_dir, style_path.split('.')[0]+'.jpg')).convert('1'))
        if (style_image.shape[0]==1):
            style_image = style_image.repeat(3,1,1)
        if (style_image.shape[0]>3):
            style_image = style_image[0:3,:,:]
        if(not style_image.shape[0]==3):
            print("Image Channel Exception",style_image.shape,content_path)

        style_image = style_image.to(device).unsqueeze(0)
        content_image = content_image.to(device).unsqueeze(0)
        style_mask = style_mask.to(device).unsqueeze(0)
        content_mask = content_mask.to(device).unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(vgg, decoder_0, decoder_1, convs, content_image, content_mask, style_image, style_mask)
        output = output.cpu()

        output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
            content_path.split('.')[0], style_path.split('.')[0], args.save_ext)
        save_image(output, str(output_name))
        print(i)
        i=i+1
