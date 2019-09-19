import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,embedding_matrix,opt):
                 #    seq_len,
                 # vocab_size,
                 # embd_size,
                 # n_layers,
                 # kernel,
                 # out_chs,
                 # res_block_count,
                 # ans_size:
        '''
        三个卷积，两个门，不共享参数：
        1、针对context的卷积，卷积时无padding，激活函数使用Relu，卷积特征中要加入aspect信息（可来自3，也可以是word embedding）
        2、针对context的卷积，卷积时无padding,激活函数使用tanh
        3、针对aspect targets的卷积，卷积时使用padding，激活函数使用tanh；
        两个激活门的输出做pairwise multiply，然后过max-pooling
        '''

        super(GatedCNN, self).__init__()
        # self.res_block_count = res_block_count
        self.opt = opt
        kernel_size = (2,embedding_matrix.shape[1])      #暂时先用一个卷积
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        # self.embedding = nn.Embedding(vocab_size, embd_size)

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv1 = nn.Conv2d(1, self.opt.out_channels,kernel_size)
        self.conv2 = nn.Conv2d(1, self.opt.out_channels,kernel_size)
        self.conv3 = nn.Conv2d(1, self.opt.out_channels,kernel_size,padding=(1,0))  # padding设置与kernel和stride有关

        self.gate0 = nn.Tanh()
        self.gate1 = nn.ReLU()

        self.pool = nn.MaxPool2d((self.opt.max_seq_len - 1,1))
        self.adapool = nn.AdaptiveMaxPool2d((1,opt.out_channels))

        # self.conv_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0))
        # self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        # self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0))
        # self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))

        # self.conv = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        # self.conv_gate = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        # self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        # self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])

        self.fc = nn.Linear(opt.out_channels, opt.polarities_dim)

    def forward(self, inputs):
        # x: (N, seq_len)
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        aspect_len = torch.max(torch.sum(aspect_indices != 0 ,dim = -1))
        aspect_indices = aspect_indices[:,:aspect_len]
        # print('text_raw_indices:',text_raw_indices.shape)
        # print('aspect_indices:',aspect_indices.shape)

        context_embedding = self.embedding(text_raw_indices).unsqueeze(1)
        aspect_embedding  = self.embedding(aspect_indices).unsqueeze(1)
        # print('context_embedding:',context_embedding.shape)
        # print('aspect_embedding:',aspect_embedding.shape)
        conv1out = self.conv1(context_embedding).squeeze().permute(0,2,1)
        conv2out = self.conv2(context_embedding).squeeze().permute(0,2,1)
        conv3out = self.conv3(aspect_embedding).permute(0,3,2,1)
        # print('conv1out', conv1out.shape)
        # print('conv2out', conv2out.shape)
        # print('conv3out', conv3out.shape)


        aspect = self.gate0(conv3out)
        aspect = self.adapool(aspect).squeeze(2)
        # print('adapooled aspect:',aspect.shape)

        context_aspect = conv1out + aspect
        # print('context_aspect',context_aspect.shape)
        
        context_aspect = self.gate1(context_aspect)
        context = self.gate0(conv2out)
        

        context_aspect = context_aspect * context
        # print(context_aspect.shape)
        # assert 1<0 
        pooled = self.pool(context_aspect).squeeze()
        out = self.fc(pooled)





        # Embedding
        # bs = x.size(0) # batch size
        # seq_len = x.size(1)
        # x = self.embedding(x) # (bs, seq_len, embd_size)

        # # CNN
        # x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # # Conv2d
        # #    Input : (bs, Cin,  Hin,  Win )
        # #    Output: (bs, Cout, Hout, Wout)
        # A = self.conv_0(x)      # (bs, Cout, seq_len, 1)
        # A += self.b_0.repeat(1, 1, seq_len, 1)
        # B = self.conv_gate_0(x) # (bs, Cout, seq_len, 1)
        # B += self.c_0.repeat(1, 1, seq_len, 1)
        # h = A * F.sigmoid(B)    # (bs, Cout, seq_len, 1)
        # res_input = h # TODO this is h1 not h0

        # for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
        #     A = conv(h) + self.b[i].repeat(1, 1, seq_len, 1)
        #     B = conv_gate(h) + self.c[i].repeat(1, 1, seq_len, 1)
        #     h = A * F.sigmoid(B) # (bs, Cout, seq_len, 1)
        #     if i % self.res_block_count == 0: # size of each residual block
        #         h += res_input
        #         res_input = h

        # h = h.view(bs, -1) # (bs, Cout*seq_len)
        # out = self.fc(h) # (bs, ans_size)
        # out = F.log_softmax(out)
        return out