import torch
import torch.nn as nn
from models.backbones import pointnet2,resnet50
from models.transformer import TransformerEncoderLayer_CMA


class CMA_fusion(nn.Module):
    def __init__(self, img_inplanes, pc_inplanes, cma_planes = 1024): #img2048   pc1024
        super(CMA_fusion, self).__init__()
        self.encoder = TransformerEncoderLayer_CMA(d_model = cma_planes, nhead = 8, dim_feedforward = 2048, dropout = 0.1)
        self.linear1 = nn.Linear(img_inplanes,cma_planes)
        self.linear2 = nn.Linear(pc_inplanes,cma_planes)
        self.quality1 = nn.Linear(cma_planes * 4, cma_planes * 2)
        self.quality2 = nn.Linear(cma_planes * 2,1)
        self.img_bn = nn.BatchNorm1d(cma_planes)
        self.pc_bn = nn.BatchNorm1d(cma_planes)         
   
    def forward(self, img, pc):
        # linear mapping and batch normalization
        #线性操作之前的img.shape: torch.Size([6, 2048])
        #线性操作之前的pc.shape: torch.Size([6, 1024])
        img = self.linear1(img)
        #Img linear1: torch.Size([6, 1024])
        img = self.img_bn(img)
        #Img bn: torch.Size([6, 1024])

        pc = self.linear2(pc)
        #PC linear1: torch.Size([6, 1024])
        pc = self.pc_bn(pc)
        #Pc bn: torch.Size([6, 1024])

        # cross modal attention and feature fusion
        img = img.unsqueeze(0)#在原有维度上插入一个新的维度
        pc = pc.unsqueeze(0)
        #print size of pc and img： torch.Size([1, 6, 1024]) torch.Size([1, 6, 1024])
        img_a,pc_a = self.encoder(img,pc)
        #torch.Size([1, 6, 1024]) torch.Size([1, 6, 1024])
        output = torch.cat((img,img_a,pc_a,pc), dim=2)
        output = output.squeeze(0)
        # feature regression
        output = self.quality1(output)
        output = self.quality2(output)
        return output

class AFF_fusion(nn.Module):
    '''
    多特征融合 AFF_fusion
    '''

    def __init__(self, img_inplanes, pc_inplanes, cma_planes=1024, channels=64, r=4):
        super(AFF_fusion, self).__init__()
        inter_channels = int(cma_planes // r)
        self.encoder = TransformerEncoderLayer_CMA(d_model=cma_planes, nhead=8, dim_feedforward=2048, dropout=0.1)

        #对齐
        self.linear1 = nn.Linear(cma_planes, cma_planes)
        self.linear2 = nn.Linear(cma_planes*2, cma_planes)

        #self.quality1 = nn.Linear(cma_planes, cma_planes // 2)
        #self.quality2 = nn.Linear(cma_planes // 2, 1)
        self.quality1 = nn.Linear(cma_planes * 2, cma_planes * 1)
        self.quality2 = nn.Linear(cma_planes, 1)

        self.img_bn = nn.BatchNorm1d(cma_planes)
        self.pc_bn = nn.BatchNorm1d(cma_planes)

        # self.img_inplanes = img_inplanes
        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Linear(cma_planes, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, cma_planes),
            nn.BatchNorm1d(cma_planes),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.Linear(cma_planes, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, cma_planes),
            nn.BatchNorm1d(cma_planes),
            nn.AdaptiveAvgPool1d(1),
        )

        self.sigmoid = nn.Sigmoid()



    def forward(self, pc, img):
        img = self.linear1(img)
        img = self.img_bn(img)

        pc = self.linear2(pc)
        pc = self.pc_bn(pc)

        img_a = img.unsqueeze(0)  # 在原有维度上插入一个新的维度
        pc_a = pc.unsqueeze(0)
        img_b, pc_b = self.encoder(img_a, pc_a)
        img_b = img_b.squeeze(0)
        pc_b = pc_b.squeeze(0)

        xa = pc + img

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = pc * wei + img * (1 - wei)
        #融合img和pc

        xa_a=pc_b+img_b

        xl_a = self.local_att(xa_a)
        xg_a = self.global_att(xa_a)
        xlg_a = xl_a+xg_a
        wei_a = self.sigmoid(xlg_a)
        xo_a = pc_b*wei_a+img_b*(1-wei_a)
        #融合img_b和pc_b


        output = torch.cat((xo,xo_a), dim=1)

        output = self.quality1(output)
        output = self.quality2(output)



        return output


class MM_PCQAnet(nn.Module):
    def __init__(self):
        super(MM_PCQAnet, self).__init__()
        self.img_inplanes = 2048
        self.pc_inplanes = 1024
        self.cma_planes=1024
        self.aaf_planes = 1024
        self.img_backbone = resnet50(pretrained=True)
        self.pc_backbone = pointnet2()
        self.regression = AFF_fusion(img_inplanes = self.img_inplanes, pc_inplanes = self.pc_inplanes, cma_planes = self.cma_planes,channels=64,r=4)
        #self.regression = CMA_fusion(img_inplanes = self.img_inplanes, pc_inplanes = self.pc_inplanes, cma_planes = self.cma_planes)
   
    def forward(self, img, pc):
        # extract features from the projections
        img_size = img.shape
        img = img.view(-1, img_size[2], img_size[3], img_size[4])
        img = self.img_backbone(img)
        img = torch.flatten(img, 1)
        # average the projection features
        img = img.view(img_size[0],img_size[1],self.img_inplanes)
        # img.shape: torch.Size([6, 4, 2048])
        img = torch.mean(img, dim = 1)
        # extract features from patches
        pc_size = pc.shape
        pc = pc.view(-1,pc_size[2],pc_size[3])
        pc = self.pc_backbone(pc)

        # average the patch features
        pc = pc.view(pc_size[0],pc_size[1],self.pc_inplanes)
        # pc.shape: torch.Size([6, 6, 1024])
        pc = torch.mean(pc, dim = 1)


        # attention, fusion, and regression
        output = self.regression(img,pc)

        return output


