import torch

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    # 创建嵌入函数
    def create_embedding_fn(self):
        embed_fns = [] # 嵌入函数列表
        d = self.kwargs['input_dims'] # 输入维度
        out_dim = 0                   # 输出维度
        if self.kwargs['include_input']: 
            embed_fns.append(lambda x: x) # 原始输入 
            out_dim += d # 输出维度+=原始输入维度

        max_freq = self.kwargs['max_freq_log2'] # 最大频率的log2值
        N_freqs = self.kwargs['num_freqs'] # 频率的数量

        if self.kwargs['log_sampling']: # 对数采样
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs) # [2^0, 2^1, ..., 2^max_freq]
        else: # 线性采样
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)
        
        # 遍历频率和周期函数，创建嵌入函数
        # embed_fns: [sin(2^0*x), cos(2^0*x), sin(2^1*x), cos(2^1*x), ..., sin(2^max_freq*x), cos(2^max_freq*x)], x.dim=3
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d # 输出维度+=嵌入向量维度

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    # 输入为3D点或视角方向，输出为嵌入向量, 嵌入向量维度=原始输入维度+频率的数量*周期函数数量*输入维度
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# 获取嵌入函数和输出维度
def get_embedder(multires):
    embed_kwargs = {
        'include_input': True, # 嵌入向量是否包含原始输入
        'input_dims': 3,       # 输入维度，3D点是(x,y,z),视角方向是(vx,vy,vz)
        'max_freq_log2': multires-1, # 最大频率的log2值
        'num_freqs': multires,       # 频率的数量
        'log_sampling': True,        # 是否采用对数采样
        'periodic_fns': [torch.sin, torch.cos], # 周期函数列表，sin和cos
    }

    embedder_obj = Embedder(**embed_kwargs) # 创建嵌入器对象
    def embed(x, eo=embedder_obj): return eo.embed(x) # 嵌入器函数，输入为3D点或视角方向，输出为嵌入向量
    return embed, embedder_obj.out_dim # 返回嵌入器函数和输出维度
