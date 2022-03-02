import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
    def forward(self, x):
        # out = self.RDB1(x)
        # out = self.RDB2(out)
        # out = self.RDB3(out)
        # return out * 0.2 + x

        out1 = self.RDB1(x)
        out2 = self.RDB2(x + 0.2 * out1)
        out3 = self.RDB3(x + 0.2 * out1 + 0.2 * out2)
        out4 = self.conv(x + 0.2 * out1 + 0.2 * out2 + 0.2 * out3)
        return out4 * 0.2 + x

class DRRDBnet(nn.Module):
    def __init__(self, nf, gc):
        super(DRRDBnet, self).__init__()
        self.DRRDB1 = RRDB(nf, gc)
        self.DRRDB2 = RRDB(nf, gc)
        self.DRRDB3 = RRDB(nf, gc)
        self.DRRDB4 = RRDB(nf, gc)
        self.DRRDB5 = RRDB(nf, gc)
        self.DRRDB6 = RRDB(nf, gc)
        self.DRRDB7 = RRDB(nf, gc)
        self.DRRDB8 = RRDB(nf, gc)
        self.DRRDB9 = RRDB(nf, gc)
        self.DRRDB10 = RRDB(nf, gc)
        self.DRRDB11 = RRDB(nf, gc)
        self.DRRDB12 = RRDB(nf, gc)
        self.DRRDB13 = RRDB(nf, gc)
        self.DRRDB14 = RRDB(nf, gc)
        self.DRRDB15 = RRDB(nf, gc)
        self.DRRDB16 = RRDB(nf, gc)
        self.DRRDB17 = RRDB(nf, gc)
        self.DRRDB18 = RRDB(nf, gc)
        self.DRRDB19 = RRDB(nf, gc)
        self.DRRDB20 = RRDB(nf, gc)
        self.DRRDB21 = RRDB(nf, gc)
        self.DRRDB22 = RRDB(nf, gc)
        self.DRRDB23 = RRDB(nf, gc)

    # def _add_(self, x, *args):
    #     result = x
    #     for a in args:
    #         result += 0.2*a
    #     return result



    def forward(self, x):
        m1 = self.DRRDB1(x)
        m2 = self.DRRDB2(x + 0.2 * m1)
        m3 = self.DRRDB3(x + 0.2 * m1 + 0.2 * m2)
        m4 = self.DRRDB4(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3)
        m5 = self.DRRDB5(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4)
        m6 = self.DRRDB6(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5)
        m7 = self.DRRDB7(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6)
        m8 = self.DRRDB8(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7)
        m9 = self.DRRDB9(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8)
        m10 = self.DRRDB10(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9)
        m11 = self.DRRDB11(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10)
        m12 = self.DRRDB12(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11)
        m13 = self.DRRDB13(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12)
        m14 = self.DRRDB14(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13)
        m15 = self.DRRDB15(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14)
        m16 = self.DRRDB16(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15)
        m17 = self.DRRDB17(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16)
        m18 = self.DRRDB18(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17)
        m19 = self.DRRDB19(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18)
        m20 = self.DRRDB20(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19)
        m21 = self.DRRDB21(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19 + 0.2 * m20)
        m22 = self.DRRDB22(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19 + 0.2 * m20 + 0.2 * m21)
        m23 = self.DRRDB23(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19 + 0.2 * m20 + 0.2 * m21 + 0.2 * m22)
        m24 = x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19 + 0.2 * m20 + 0.2 * m21 + 0.2 * m22 + 0.2 * m23
        return m24
        # return self._add_(x, m1, m2,m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23)


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb=1, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(DRRDBnet, nf=nf, gc=gc)


        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        # self.RRDB_trunk = RRDB_block_f
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='     ')))
        # fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
