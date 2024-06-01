import argparse
import random
import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms
from paddleseg.utils import *
from paddleseg.cvlibs import Config
from paddleseg.core import infer
from utils import *
from differential_color_functions import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)

# parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda', type=bool,default=True, help='enables cuda')

parser.add_argument('--max_count', type=int, default=2000, help='max number of iterations to find adversarial example')

parser.add_argument('--patch_size', type=float, default=0.01, help='The total patch size. E.g. 0.01 ~= 1% of image ')
parser.add_argument('--patch_number', type=float, default=16, help='The number of patches')

parser.add_argument('--train_size', type=int, default=7000, help='Number of training images')
parser.add_argument('--test_size', type=int, default=3000, help='Number of test images')

parser.add_argument('--image_size', type=int, default=600, help='The height / width of the input image to network')

parser.add_argument('--netClassifier', default='resnet50', help="The target classifier: resnet50/resnet34/resnet101/densenet121")

parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')

parser.add_argument('--save_path', type=str, default=r'.\new_logs_AID', help='The path to save the result')
parser.add_argument('--data_path', type=str, default=r'.\AID_test', help='The data path')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


max_count = opt.max_count
patch_size = opt.patch_size
patch_num = opt.patch_number
image_size = opt.image_size
train_size = opt.train_size
test_size = opt.test_size
patch_size = patch_size/patch_num
classifier = opt.netClassifier
save_path = opt.save_path
data_path = opt.data_path
assert train_size + test_size <= 10000, "Traing set size + Test set size > Total dataset size"

print("=> creating model ")

if(classifier == 'resnet50'):
    netClassifier = models.resnet50()
    netClassifier = torch.load(r'.\models\AID_best_9617.pt')

if(classifier == 'resnet34'):
    netClassifier = models.resnet34()
    netClassifier = torch.load(r'.\models\AID_best9510_resnet34.pt')

if (classifier == 'resnet101'):
    netClassifier = models.resnet101()
    netClassifier = torch.load(r'.\models\AID_best_9557_resnet101.pt')

if(classifier == 'densenet121'):
    netClassifier = models.densenet121()
    netClassifier = torch.load(r'.\models\AID_best_9620_densenet121.pt')

netClassifier = nn.Sequential(
    # normalize,
    netClassifier
).to(device)
netClassifier = netClassifier.eval()

cfg = Config('ocrnet_dlrsd.yml')
seq_model = cfg.model
seqmodel_path = r'.\models\model.pdparams'
load_entire_model(seq_model,seqmodel_path)
seq_model.eval()
infer_tran = cfg.val_dataset.transforms

allow_patch = [2,7,8,9,11,12]

if opt.cuda:
    netClassifier.cuda()

print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor()
])

idx = np.arange(3000)
np.random.shuffle(idx)
training_idx = idx[:train_size]
test_idx = idx[train_size:test_size]

train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(data_path, transforms.Compose([
        transforms.ToTensor(),

    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(training_idx),
    num_workers=opt.workers, pin_memory=True)


min_out, max_out = 0,1.0

def train( patch, patch_shape):
    netClassifier.eval()
    total = 0
    right_num = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for batch_idx, (data, labels) in enumerate(train_loader):

        data = data.cuda()
        labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        ori_shape = (image_size,image_size)
        data_np = data.data.cpu().numpy()
        data_np = data_np.squeeze(0)
        data_np = np.moveaxis(data_np, 0, -1)
        im, _ = infer_tran(data_np*255.0)
        im = im[np.newaxis, ...]
        im = paddle.to_tensor(im)
        pred = infer.inference(
            seq_model,
            im,
            ori_shape=ori_shape,
            transforms=infer_tran.transforms,
            is_slide=False,
        )

        prediction = netClassifier(data)
        _, pre = torch.max(prediction.data, 1)
        _, indices = torch.sort(prediction, descending=True)
        x_label = prediction.data.max(1)[1][0]
        # only computer adversarial examples on examples that are originally classified correctly
        if prediction.data.max(1)[1][0] != labels.data[0] :
            continue

        total += 1

        # transform path
        data_shape = data.data.cpu().numpy().shape

        adv_x,is_ok,is_allow = attack(data, patch, data_shape,patch_shape,image_size,x_label,pred)
        out = netClassifier(adv_x)
        _, adv_label = out.max(1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        perc = percentage.max()


        if is_ok:
            save_image(adv_x ,save_path+'\\' +str(batch_idx)+'_' + str(int(x_label))+'_' +str(int(adv_label))+'_' +str(int(perc))+'_'+ 'adv.png')
            right_num += 1
            print("Number " + str(batch_idx) + " attack success!")
        else :
            if is_allow:
                save_image(adv_x , save_path+'\\' + str(batch_idx) +'_' + 'wrong_adv.png')
                print("Number " + str(batch_idx) + " attack fail")
            else:
                save_image(adv_x, save_path+'\\' + str(batch_idx) + '_' + 'wrong_notallow_adv.png')
                print("Number " + str(batch_idx) + " not enough space to attack")
    asr = right_num / total
    print("The Attacking Success Rate is " + str(asr))

def attack(x, patch, data_shape,patch_shape,image_size,x_label,pred):
    netClassifier.eval()
    x.requires_grad = True
    out = netClassifier(x)
    x_perc = out[0][int(x_label)]
    loss1 = x_perc
    loss1.backward()
    step = x.grad.clone()
    x.grad.data.zero_()

    for i in range(3):
        x_patch, mask, is_allow = square_transform_mul_grad_seq_ram(patch, data_shape, patch_shape, image_size, patch_num,
                                                                step, pred,
                                                                allow_patch)

        x_patch, mask = torch.FloatTensor(x_patch), torch.FloatTensor(mask)

        x_patch, mask = x_patch.cuda(), mask.cuda()

        if not is_allow:
            adv_x = torch.mul((1 - mask), x) + torch.mul(mask, x_patch)
            adv_x = torch.clamp(adv_x, min_out, max_out)
            is_ok = False
            break
        else:
            x_patch = Variable(x_patch.data, requires_grad=True)
            optim = torch.optim.Adam([x_patch], lr=2 / 255,weight_decay=0.2/255)
            weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, 200, gamma=0.9)
            adv_x = torch.mul((1 - mask), x) + torch.mul(mask, x_patch)
            adv_x = torch.clamp(adv_x, min_out, max_out)

            out = netClassifier(adv_x)

            _, pre = torch.max(out.data, 1)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

            x_perc = out[0][int(x_label)]
            x_perc_t = percentage[int(x_label)]

            _, adv_label = out.max(1)

            count = 0
            is_ok = False
            while x_perc_t > 10 :
                count += 1
                optim.zero_grad()


                Loss = x_perc
                Loss.backward()
                optim.step()
                weight_scheduler.step()

                adv_x = torch.mul((1 - mask), x) + torch.mul(mask, x_patch)

                adv_x = torch.clamp(adv_x, min_out, max_out)

                out = netClassifier(adv_x)
                _, pre = torch.max(out.data, 1)
                percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

                x_perc = out[0][int(x_label)]
                x_perc_t = percentage[int(x_label)]
                _, adv_label = out.max(1)

                #print(Loss)
                #print(int(x_perc_t))

                if count >= opt.max_count:
                    # is_ok = False
                    break
                if count >= 200 and x_perc_t >= 95:
                    # print('break')
                    break

            if adv_label != x_label:
                is_ok = True
                break

    return adv_x,is_ok,is_allow


if __name__ == '__main__':
    patch, patch_shape = init_patch_square_mul(image_size, patch_size,patch_num)
    train( patch, patch_shape)

