import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import os
import itertools


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
    feat_var_1 = torch.pow(feat_var_0, 2)
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

    def forward(self, content, cm, style, sm):
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
                g_t_i_1 = self.decoder_1(t_l)  
            else:
                g_t_i_1 = g_t_i_0

            g_t_i = torch.cat((g_t_i_0, g_t_i_1), dim=1)
            g_t_i = self.convs(g_t_i)
            g_t.append(g_t_i) 
            g_t_feats = self.encode_with_intermediate(g_t_i)

            loss_c += self.calc_content_loss(g_t_feats[-1], content_feat[i,:,:,:])
            loss_s += self.calc_style_loss(g_t_feats[0], style_feats[0][i,:,:,:].unsqueeze(0))
            for q in range(1, 4):
                loss_s += self.calc_style_loss(g_t_feats[q], style_feats[q][i,:,:,:].unsqueeze(0))
        
        loss_c = loss_c / 8
        loss_s = loss_s / 8
        
        return g_t, loss_c, loss_s


def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31 


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize([256,256]),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, root_mask, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root 
        self.root_mask = root_mask 
        self.paths = os.listdir(root_mask)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        im_name = os.path.join(self.root, path.split('.')[0]+'.jpg')
        mask_name = os.path.join(self.root_mask, path)
        img = Image.open(im_name).convert('RGB')
        mask = Image.open(mask_name).convert('1')
        img = self.transform(img)
        mask = self.transform(mask)
        item = {'img': img, 'mask':mask ,'name': path} #put the image and its name together into item(dict)
        return item

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='./image/trainC',
                    help='Directory path to content images')
parser.add_argument('--style_dir', type=str, default='./image/trainS',
                    help='Directory path to style images')
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')
parser.add_argument('--content_mask_dir', type=str, default='./mask/trainC',
                    help='Directory path to content masks')
parser.add_argument('--style_mask_dir', type=str, default='./mask/trainS',
                    help='Directory path to style masks')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./log_train',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder_0 = decoder_0
decoder_1 = decoder_1
convs = convs
vgg = vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = Net(vgg, decoder_0, decoder_1, convs)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, args.content_mask_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, args.style_mask_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(itertools.chain(network.decoder_0.parameters(),network.decoder_1.parameters(),network.convs.parameters()), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content = next(content_iter)
    style = next(style_iter)
    content_image = content['img'].cuda()
    content_mask = content['mask'].cuda()
    style_image = style['img'].cuda()
    style_mask = style['mask'].cuda()
    g_t, loss_c, loss_s = network(content_image, content_mask, style_image, style_mask)

    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
   
    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_image('content_image', content_image[0,:,:,:],i+1)
    writer.add_image('content_mask', content_mask[0,:,:,:],i+1)
    writer.add_image('style_image', style_image[0,:,:,:],i+1)
    writer.add_image('style_mask', style_mask[0,:,:,:],i+1)
    writer.add_image('g_t', g_t[0].squeeze(),i+1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict_0 = decoder_0.state_dict()
        state_dict_1 = decoder_1.state_dict()
        state_dict = convs.state_dict()
        for key in state_dict_0.keys():
            state_dict_0[key] = state_dict_0[key].to(torch.device('cpu'))
        torch.save(state_dict_0, save_dir /
                   'decoder_0_iter_{:d}.pth.tar'.format(i + 1))
        for key in state_dict_1.keys():
            state_dict_1[key] = state_dict_1[key].to(torch.device('cpu'))
        torch.save(state_dict_1, save_dir /
                   'decoder_1_iter_{:d}.pth.tar'.format(i + 1))
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'convs_iter_{:d}.pth.tar'.format(i + 1))

writer.close()


