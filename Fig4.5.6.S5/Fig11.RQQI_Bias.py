# -*- coding: utf-8 -*-
import os
from osgeo import osr
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import math
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import datetime
from matplotlib import rcParams


def listdir(path):
    list_name = []
    for file in os.listdir(path):
        # print(os.path.splitext(file)[0][-11:])
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    return list_name


def MODISlistdir(path):
    list_name = []
    for file in os.listdir(path):
        # print(os.path.splitext(file)[0][-11:])
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, file[0:33])
            if file_path in list_name:
                continue
            list_name.append(file_path)
    return list_name


def VIIRSlistdir(path):
    # clipprojectVNP15A2H.A2012017.h12v04.Fpar.tif
    list_name = []
    for file in os.listdir(path):
        # print(os.path.splitext(file)[0][-11:])
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, file[0:36])
            if file_path in list_name:
                continue
            list_name.append(file_path)
    return list_name


base = [str(x) for x in range(10)] + [chr(x) for x in range(ord('A'), ord('A') + 6)]


def dec2bin8(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num, rem = divmod(num, 2)
        mid.append(base[rem])

    result = ''.join([str(x) for x in mid[::-1]])
    for i in range(len(result), 8):
        result = "0" + result
    return result


def dec2bin16(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num, rem = divmod(num, 2)
        mid.append(base[rem])

    result = ''.join([str(x) for x in mid[::-1]])
    for i in range(len(result), 16):
        result = "0" + result
    return result


def dec2bin(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num, rem = divmod(num, 2)
        mid.append(base[rem])
    result = ''.join([str(x) for x in mid[::-1]])
    for i in range(len(result), 8):
        result = "0" + result
    return result


if __name__ == "__main__":
    s=os.sep
    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    cdict = []
    for x in range(0, 250):
        hsv = ((250 - x) / 360.0, 1, 1)
        rgb = colors.hsv_to_rgb(hsv)
        cdict.append(rgb)
    cm = colors.ListedColormap(cdict, 'zyh')

    fontsize = 20
    font = {'family': 'Times New Roman',
            'color': 'k',
            'weight': 'normal',
            'size': fontsize, }
    legendfont = {'family': 'Times New Roman',
                  'size': fontsize}
    ##
    fig = plt.figure(figsize=(12, 15), dpi=300)
    sitelist = ["CA-TP4", "CA-TPD", "US-Bar", "US-HF"]
    subi=1
    for eachdata in ["MOD", "MYD", "VNP", "PROBAV300", "PROBAV1000"]:
        for eachsite in sitelist:
            # subi=subi+1
            ALLX = []
            ALLY = []
            ALLDOY = []
            if eachdata in ["MOD", "MYD", "VNP"]:
                TPDdf = pd.read_csv(r".\Data" + s + eachsite + "_HLS_MODIS_FAPAR_RMSE_QC.csv")
                ncols = TPDdf.shape[1]
                TPDnpixels = (ncols - 3) // 10
            else:
                TPDdf = pd.read_csv(r".\Data" + s + eachsite + "_HLS_" + eachdata + "_FAPAR_RMSE_QC.csv")
                ncols = TPDdf.shape[1]
                TPDnpixels = (ncols - 3) // 4

            for ipixel in range(1, TPDnpixels + 1):
                tempTPDdf = TPDdf[(TPDdf[eachdata + "FAPAR" + str(ipixel)] > 0) & (TPDdf[eachdata + "STD" + str(ipixel)] > 0)& (TPDdf[eachdata + "STD" + str(ipixel)] <1)]
                tempZ = tempTPDdf["doy"]
                tempHLS = tempTPDdf["HLS" + str(ipixel)]
                tempPRO = tempTPDdf[eachdata + "FAPAR" + str(ipixel)]
                tempYdf=tempPRO-tempHLS
                tempY=tempYdf.abs()
                tempX=tempTPDdf[eachdata + "STD" + str(ipixel)]/tempTPDdf[eachdata + "FAPAR" + str(ipixel)]
                ALLX.extend(tempX)
                ALLY.extend(tempY)
                ALLDOY.extend(tempZ)
            df=pd.DataFrame()
            df["X"]=ALLX
            df["Y"]=ALLY


            ax = fig.add_subplot(5,4,subi)

            if subi <6:
                ax.text(0.02, 1.1, eachsite, verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize)

            ax.text(0.02, 0.98, "(" + chr(96 + subi) + ")", verticalalignment='top',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize)


            sc = ax.scatter(ALLX, ALLY, marker="o", s=20, linewidths=0.2, edgecolors='k', c=ALLDOY, vmin=1, vmax=360,
                            cmap=cm)

            # N = len(ALLY)
            #
            x1 = np.array(ALLX).reshape(-1, 1)
            y1 = np.array(ALLY).reshape(-1, 1)
            regr1 = linear_model.LinearRegression()
            regr1.fit(x1, y1)
            yy1 = regr1.predict(x1)
            r21 = round(regr1.score(x1, y1), 2)
            a1 = round(regr1.coef_[0][0], 2)
            b1 = round(regr1.intercept_[0], 2)
            rmse1 = round(math.sqrt(mean_squared_error(x1, y1)), 2)
            bias1 = (round(np.mean(y1 - x1), 2))


            ax.text(0.97, 0.75, r'R$^2$=' + str(r21), verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize)

            if b1 > 0:
                ax.text(0.97, 0.87, r'y=' + str(a1) + "x+" + str(b1), verticalalignment='bottom',
                        horizontalalignment='right',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize)
            else:
                ax.text(0.97, 0.87, r'y=' + str(a1) + "x$-$" + str(abs(b1)), verticalalignment='bottom',
                        horizontalalignment='right',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize)
            # print(eachSite + eachProduct + " " + str(r21) + " " + str(bias1) + " " + str(rmse1) + " " + str(
            #     round(rrmse * 100, 1)) + " " + str(PGCOS))


            subi = subi + 1
            # 拟合线
            ax.plot(x1, yy1, color='k', linewidth=1.2)
            # y=x线

            ax.axis([0, 0.35, 0, 0.8])

            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            #
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(fontsize) for label in labels]

            if subi >17:
                ax.set_xlabel("Relative QQIs", fontdict=font)
            else:
                ax.set_xticklabels([])
            # [2, 7, 12, 17, 22]
            if subi in [2]:
                ax.set_ylabel("MOD", fontdict=font)
            elif subi in [6]:
                ax.set_ylabel("MYD", fontdict=font)
            elif subi in [10]:
                ax.set_ylabel("VNP", fontdict=font)
            elif subi in [14]:
                ax.set_ylabel("GEOV3", fontdict=font)
            elif subi in [18]:
                ax.set_ylabel("GEOV2", fontdict=font)
            else:
                ax.set_yticklabels([])

            if subi in [5,9,13,17,21]:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                # plt.colorbar(im, cax=cax)
                cbar = plt.colorbar(sc, cax=cax)
                # cbar.set_label('DOY',fontdict=font)
                # cbar.set_ticks(np.linspace(160, 260, 6))
                cbar.ax.tick_params(labelsize=fontsize)
                labels = cbar.ax.get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]

    # plt.subplots_adjust(left=0.13, right=0.96, bottom=0.08, top=0.98, wspace=0.35, hspace=0.35)
    plt.subplots_adjust(left=0.07, right=0.95, bottom=0.05, top=0.98, wspace=0.1, hspace=0.1)

    plt.savefig('Fig11.RQQI_Bias.png')  # 保存图片
