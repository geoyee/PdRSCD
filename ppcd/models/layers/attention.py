import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CAM(nn.Layer):
    # Channels Attention Module
    def __init__(self, in_channels, ratio=8):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.mlp = nn.Sequential(
            nn.Conv2D(in_channels, in_channels//ratio, 1),
            nn.ReLU(),
            nn.Conv2D(in_channels//ratio, in_channels, 1)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = F.sigmoid(avg_out + max_out)
        return out


class SAM(nn.Layer):
    # Spatial Attention Module
    def __init__(self):
        super(SAM, self).__init__()
        self.conv = nn.Conv2D(2, 1, 7, padding=3)

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv(x)
        x = F.sigmoid(x)
        return x


class BAM(nn.Layer):
    """ 
        Basic self-attention module
    """
    def __init__(self, in_channels, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.key_channel = in_channels //8
        self.activation = activation
        self.ds = ds
        self.pool = nn.AvgPool2D(self.ds)
        self.query_conv = nn.Conv2D(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2D(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.ParameterList([paddle.create_parameter(shape=[1], dtype='float32', default_initializer=nn.initializer.Constant(value=0))])
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, input):
        """
            inputs :
                x : input feature maps(B C W H)
            returns :
                out : self attention value + input feature
                attention: B N N (N is Width*Height)
        """
        x = self.pool(input)
        N, C, H, W = x.shape
        proj_query = self.query_conv(x).reshape([N, -1, H * W]).transpose((0, 2, 1))
        proj_key = self.key_conv(x).reshape([N, -1, H * W])
        energy = paddle.bmm(proj_query, proj_key)
        energy = (self.key_channel ** -.5) * energy
        attention = self.softmax(energy - paddle.max(energy, axis=-1, keepdim=True))  # 防止溢出
        proj_value = self.value_conv(x).reshape([N, -1, H * W])
        out = paddle.bmm(proj_value, attention.transpose((0, 2, 1)))
        out = out.reshape([N, C, H, W])
        out = F.interpolate(out, [H * self.ds, W * self.ds])
        out = out + input
        return out


class _PAMBlock(nn.Layer):
    '''
        The basic implementation for self-attention block/non-local block
        Input/Output:
            N C H (2*W)
        Parameters:
            in_channels       : the dimension of the input feature map
            key_channels      : the dimension after the key/query transform
            value_channels    : the dimension after the value transform
            scale             : choose the scale to partition the input feature maps
            ds                : downsampling scale
    '''
    def __init__(self, in_channels, key_channels, value_channels, scale=1, ds=1):
        super(_PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2D(self.ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.f_key = nn.Sequential(
            nn.Conv2D(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(self.key_channels)
        )
        self.f_query = nn.Sequential(
            nn.Conv2D(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(self.key_channels)
        )
        self.f_value = nn.Conv2D(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x = input
        if self.ds != 1:
            x = self.pool(input)
        batch_size, _, h, w = x.shape[0], x.shape[2], x.shape[3] // 2
        local_y = []
        local_x = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale-1):
                    end_x = h
                if j == (self.scale-1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)
        value = paddle.stack([value[:, :, :, :w], value[:,:,:,w:]], 4)
        query = paddle.stack([query[:, :, :, :w], query[:,:,:,w:]], 4)
        key = paddle.stack([key[:, :, :, :w], key[:,:,:,w:]], 4)
        local_block_cnt = 2 * self.scale * self.scale

        #  self-attention func
        def func(value_local, query_local, key_local):
            batch_size_new = value_local.shape[0]
            h_local, w_local = value_local.shape[2], value_local.shape[3]
            value_local = value_local.reshape([batch_size_new, self.value_channels, -1])
            query_local = query_local.reshape([batch_size_new, self.key_channels, -1])
            query_local = query_local.transpose((0, 2, 1))
            key_local = key_local.reshape([batch_size_new, self.key_channels, -1])
            sim_map = paddle.bmm(query_local, key_local)
            sim_map = (self.key_channels ** -.5) * sim_map
            attention = F.softmax(sim_map - paddle.max(sim_map, axis=-1, keepdim=True), axis=-1)
            context_local = paddle.bmm(value_local, attention.transpose((0, 2, 1)))
            context_local = context_local.reshape([batch_size_new, self.value_channels, h_local, w_local, 2])
            return context_local

        #  Parallel Computing to speed up
        #  reshape value_local, q, k
        v_list = [value[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        v_locals = paddle.concat(v_list, axis=0)
        q_list = [query[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        q_locals = paddle.concat(q_list, axis=0)
        k_list = [key[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        k_locals = paddle.concat(k_list, axis=0)
        context_locals = func(v_locals,q_locals,k_locals)
        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = batch_size * (j + i * self.scale)
                right = batch_size * (j + i * self.scale) + batch_size
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(paddle.concat(row_tmp, 3))
        context = paddle.concat(context_list, 2)
        context = paddle.concat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)
        if self.ds !=1:
            context = F.interpolate(context, [h * self.ds, 2 * w * self.ds])
        return context


class PAMBlock(_PAMBlock):
    def __init__(self, in_channels, key_channels=None, value_channels=None, scale=1, ds=1):
        if key_channels == None:
            key_channels = in_channels // 8
        if value_channels == None:
            value_channels = in_channels
        super(PAMBlock, self).__init__(in_channels,key_channels,value_channels,scale,ds)


class PAM(nn.Layer):
    """
        PAM module
    """
    def __init__(self, in_channels, out_channels, sizes=([1]), ds=1):
        super(PAM, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.ds = ds  # output stride
        self.value_channels = out_channels
        self.key_channels = out_channels // 8
        self.stages = nn.LayerList(
            [self._make_stage(in_channels, self.key_channels, self.value_channels, size, self.ds) for size in sizes])
        self.conv_bn = nn.Sequential(
            nn.Conv2D(in_channels * self.group, out_channels, kernel_size=1, padding=0),
            # nn.BatchNorm2D(out_channels),
        )

    def _make_stage(self, in_channels, key_channels, value_channels, size, ds):
        return PAMBlock(in_channels, key_channels, value_channels, size, ds)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        #  concat
        context = []
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn(paddle.concat(context, 1))
        return output


class GatedAttentionLayer(nn.Layer):
    def __init__(self, in_channels, attention_channels):
        super(GatedAttentionLayer, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(in_channels, attention_channels),  # V
            nn.Tanh(),  # tanh(V * H_t)
            nn.Linear(attention_channels, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(in_channels, attention_channels),  # U
            nn.Sigmoid()  # sigm(U * H_t)
        )
        # W_t * [tanh(V * H_t) * sigm(U * H_t)]
        self.w_t = nn.Linear(attention_channels, 1)

    def forward(self, x):
        a1 = self.att(x)
        a2 = self.gate(x)
        a3 = a1 * a2
        return self.w_t(a3)