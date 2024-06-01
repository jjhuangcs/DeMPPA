import os
import sys
import time
import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from scipy.ndimage.interpolation import rotate

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 80
TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' ' + msg)
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements, 
    # we can find the desired rectangular bounds.  
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]


class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr
    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor




def init_patch_square(image_size, patch_size):
    # get mask
    image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size**(0.5))
    patch = np.random.rand(1,3,noise_dim,noise_dim)
    return patch, patch.shape

def init_patch_square_mul(image_size, patch_size,patch_num):
    # get mask
    image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = math.ceil(noise_size**(0.5))
    patch = np.random.randn(patch_num,3,noise_dim,noise_dim)
    return patch, patch.shape



def percentile(x):
    return x / np.sum(x, axis=0)



def softmax( f ):
    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f) # f becomes [-666, -333, 0]
    return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer

def square_transform_mul_grad_seq_ram(patch, data_shape, patch_shape, image_size,patch_num,step,pred,allow_patch):
    # get dummy image
    x = np.zeros(data_shape)
    mask = np.copy(x)
    # get shape
    m_size = patch_shape[-1]
    step = abs(step)
    step = step[0][0] + step[0][1] + step[0][2]
    step = step.unsqueeze(0)
    step_tem = step.clone()
    cc = torch.nn.Conv2d(1, 1, m_size, stride=1, padding=0, bias=False)
    cc.weight.data = torch.ones(1, 1, m_size, m_size).cuda()
    patch_x = {}
    patch_y = {}
    type = pred[0][0].numpy()

    for allow in allow_patch:
        type[type == allow] = -1

    type[type != -1] = 0
    type[type == -1] = 1
    type_m = torch.from_numpy(type).float().cuda()
    type_m = type_m.unsqueeze(0)
    allow_xy = cc(type_m)
    allow_a, allow_b = torch.where(allow_xy[0] == m_size * m_size)
    allow_xy[allow_xy != m_size * m_size] = 0
    allow_xy[allow_xy == m_size * m_size] = 1
    is_allow = True
    if len(allow_a) < patch_num * m_size * m_size:
        is_allow = False
    if is_allow:
        x = np.zeros(data_shape)
        for i in range(patch_num):

            # random location
            grad_size = image_size - m_size + 1
            grad = cc(step_tem)
            grad = torch.mul(grad,allow_xy)
            grad_np = grad.data.cpu().numpy()
            grad_1 = grad_np.flatten()
            grad_2 = softmax(grad_1.copy()/10)
            index = np.random.choice(range(grad_size * grad_size), 1, p=grad_2)

            patch_x[i] = int(index / grad_size)
            patch_y[i] = int(index - patch_x[i] * grad_size)
            # apply patch to dummy image
            x[0][0][patch_x[i]:patch_x[i] + patch_shape[-1], patch_y[i]:patch_y[i] + patch_shape[-1]] = patch[i][0]
            x[0][1][patch_x[i]:patch_x[i] + patch_shape[-1], patch_y[i]:patch_y[i] + patch_shape[-1]] = patch[i][1]
            x[0][2][patch_x[i]:patch_x[i] + patch_shape[-1], patch_y[i]:patch_y[i] + patch_shape[-1]] = patch[i][2]
            step_tem[0][patch_x[i]:patch_x[i] + patch_shape[-1], patch_y[i]:patch_y[i] + patch_shape[-1]] = 0

        mask = np.copy(x)
        mask[mask != 0] = 1.0

    return x, mask,is_allow



def square_transform_mul_recover(patch, data_shape, patch_shape, image_size,patch_num):
    # get dummy image
    x = np.zeros(data_shape)
    mask = np.copy(x)
    # get shape
    m_size = patch_shape[-1]

    random_x = {}
    random_y = {}
    x = np.zeros(data_shape)
    i = 0
    for i in range(patch_num):
        # random rotation
        rot = np.random.choice(4)
        k = 0
        for k in range(patch[i].shape[0]):
            patch[i][k] = np.rot90(patch[i][k], rot)

        # random location
        random_x[i] = np.random.choice(image_size)
        if random_x[i] + m_size > x.shape[-1]:
            while random_x[i] + m_size > x.shape[-1]:
                random_x[i] = np.random.choice(image_size)
        random_y[i] = np.random.choice(image_size)
        if random_y[i] + m_size > x.shape[-1]:
            while random_y[i] + m_size > x.shape[-1]:
                random_y[i] = np.random.choice(image_size)

        # apply patch to dummy image
        x[0][0][random_x[i]:random_x[i] + patch_shape[-1], random_y[i]:random_y[i] + patch_shape[-1]] = patch[i][0]
        x[0][1][random_x[i]:random_x[i] + patch_shape[-1], random_y[i]:random_y[i] + patch_shape[-1]] = patch[i][1]
        x[0][2][random_x[i]:random_x[i] + patch_shape[-1], random_y[i]:random_y[i] + patch_shape[-1]] = patch[i][2]

    mask = np.copy(x)
    mask[mask != 0] = 1.0


    return x, mask

def get_patch(patch,mask,patch_num,patch_shape):
    p = np.zeros(patch_shape)
    patch_data = patch.data.cpu().numpy()
    patch_size = patch_shape[-1]
    mask_tem = mask.clone()
    x, y = np.nonzero(mask[0][0].data.cpu().numpy())
    i=0
    for j in range(x.min(),x.max()):
        if i > patch_num-1 :
            break
        for k in range(y.min(),y.max()):
            if i > patch_num-1:
                break
            if mask_tem[0][0][j][k] != 0:
                p[i][0] = patch_data[0][0][j:j + patch_size, k:k + patch_size]
                p[i][1] = patch_data[0][1][j:j + patch_size, k:k + patch_size]
                p[i][2] = patch_data[0][2][j:j + patch_size, k:k + patch_size]
                mask_tem[0][0][j:j + patch_size, k:k + patch_size] = 0
                i += 1

    return p
def is_overlap(mask,patch_num,patch_size):
    x,y = np.nonzero(mask[0][0])
    if len(x) != patch_num*patch_size*patch_size:
        return True
    else:
        return False


def get_patch_1(patch,mask,patch_num,patch_shape):
    p = np.zeros(patch_shape)
    patch_data = patch.data.cpu().numpy()
    patch_size = patch_shape[-1]
    mask_tem = mask.clone()
    i=0
    while i < patch_num:
        x, y = np.nonzero(mask_tem[0][0].data.cpu().numpy())
        is_find = False
        for j in range(x.min(), x.max()):
            if not is_find:
                for k in range(y.min(), y.max()):
                    if mask_tem[0][0][j][k] != 0:
                        p[i][0] = patch_data[0][0][j:j + patch_size, k:k + patch_size]
                        p[i][1] = patch_data[0][1][j:j + patch_size, k:k + patch_size]
                        p[i][2] = patch_data[0][2][j:j + patch_size, k:k + patch_size]
                        mask_tem[0][0][j:j + patch_size, k:k + patch_size] = 0
                        i += 1
                        is_find = True
                        break
            else:
                break

    return p


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

def tv_loss(x):

    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)

def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),
                                               requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def gauss_noise(img, mean=0, sigma=25):
    image = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, sigma / 255.0, image.shape)
    out = image + noise
    out = np.clip(out, 0.0, 1.0)
    # res_img = np.uint8(res_img * 255.0)

    return out

