import os
from glob import glob
import torch

# 创建目录
def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

# 根据指定路径获取类
def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

# 获取指定路径下的所有图像文件路径
def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

# 分割输入，适配cuda显存，避免出现OOM问题
def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000  # 每1w个像素点作为一个数据块(chunk)
    split = []        # 分割好的数据
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx) # 这里的model_input['uv'].shape是[batch_size, H*W, 2]
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx) # 这里的model_input['object_mask'].shape是[batch_size, H*W]
        split.append(data) # 放入list
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''
    # 将多个split的模型输出结果合并成一个结果字典
    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs