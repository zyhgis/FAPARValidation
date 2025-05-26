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
    for eachdata in ["MOD","MYD","VNP","PROBAV300","PROBAV1000"]:
        for eachsite in sitelist:
            outdf = pd.DataFrame()
            idlist, Nlist, Rlist, SDBlist, RSDBlist, MBlist, RMBlist, RMSElist, RRMSElist, URAoptlist, URAtarlist, URAthrlist = [], [], [], [], [], [], [], [], [], [], [], []

            # subi=subi+1
            ALLX = []
            ALLY = []
            ALLDOY = []
            if eachdata in ["MOD","MYD","VNP"]:
                TPDdf = pd.read_csv(r".\Data"+s+eachsite+"_HLS_MODIS_FAPAR_RMSE_QC.csv")
                ncols = TPDdf.shape[1]
                TPDnpixels = (ncols - 3) // 10
            else:
                TPDdf = pd.read_csv(r".\Data"+s+eachsite+"_HLS_"+eachdata+"_FAPAR_RMSE_QC.csv")
                ncols = TPDdf.shape[1]
                TPDnpixels = (ncols - 3) // 4

            for items, row in TPDdf.iterrows():
                tempFAPARpro = []
                tempFAPARhls=[]
                tenpdoy = []
                for ipixel in range(1, TPDnpixels + 1):
                    tempFAPARpro.append(row[eachdata + "FAPAR" + str(ipixel)])
                    tempFAPARhls.append(row["HLS" + str(ipixel)])

                tempFAPARhls=np.array(tempFAPARhls)
                tempFAPARpro=np.array(tempFAPARpro)
                tempFAPARhls2=tempFAPARhls[tempFAPARhls>0]
                tempFAPARpro2=tempFAPARpro[tempFAPARhls>0]
                tempFAPARhls3=tempFAPARhls2[tempFAPARpro2>0]
                tempFAPARpro3=tempFAPARpro2[tempFAPARpro2>0]




                # if len(tempFAPARhls3)/TPDnpixels<0.7:
                #     continue
                if len(tempFAPARhls3)==0:
                    continue
                ALLX.append(np.mean(tempFAPARhls3))
                ALLY.append(np.mean(tempFAPARpro3))
                ALLDOY.append(row['doy'])

            ax = fig.add_subplot(5,4,subi)
            Targetlowxi, Targetlowyi, Targetupxi, Targetupyi, Optimallowxi, Optimallowyi, Optimalupxi, Optimalupyi, Thresholdlowxi, Thresholdlowyi, Thresholdupxi, Thresholdupyi = [], [], [], [], [], [], [], [], [], [], [], []
            for ii in range(120):
                Optimallowxi.append(ii * 0.01)
                Optimallowyi.append(ii * 0.01 * 0.95)
                Optimalupxi.append(ii * 0.01)
                Optimalupyi.append(ii * 0.01 * 1.05)
                Thresholdlowxi.append(ii * 0.01)
                Thresholdlowyi.append(ii * 0.01 * 0.8)
                Thresholdupxi.append(ii * 0.01)
                Thresholdupyi.append(ii * 0.01 * 1.2)
                if ii > 50:
                    Targetupxi.append(ii * 0.01)
                    Targetupyi.append(ii * 0.01 * 1.1)
                    Targetlowxi.append(ii * 0.01)
                    Targetlowyi.append(ii * 0.01 * 0.9)
                else:
                    Targetupxi.append(ii * 0.01)
                    Targetupyi.append(ii * 0.01 + 0.05)
                    Targetlowxi.append(ii * 0.01)
                    Targetlowyi.append(ii * 0.01 - 0.05)
            ax.fill_between(Thresholdlowxi, np.array(Thresholdlowyi), np.array(Thresholdupyi), facecolor="dimgray",
                            alpha=1)
            ax.fill_between(Targetlowxi, np.array(Targetlowyi), np.array(Targetupyi), facecolor="darkgray", alpha=1)
            ax.fill_between(Optimallowxi, np.array(Optimallowyi), np.array(Optimalupyi), facecolor="lightgray", alpha=1)
            if subi <5:
                ax.text(0.02, 1.1, eachsite, verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize)

            # ax.text(0.02, 0.98, "(" + chr(96 + subi) + ")", verticalalignment='top',
            #             horizontalalignment='left',
            #             transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize)


            sc = ax.scatter(ALLX, ALLY, marker="o", s=80, linewidths=0.3, edgecolors='k', c=ALLDOY, vmin=1, vmax=365,
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
            #
            SDbias1 = round(np.std(y1 - x1), 2)
            rrmse = round(rmse1 / np.mean(ALLX), 3)
            # PGCOS = str(round(OGCOS * 1.0 / N * 100, 1))
            PGCOS = str(len(ALLX))
            N=len(ALLX)

            Nopt = 0
            Ntar = 0
            Nthr = 0
            for Ni in range(N):
                # print(Y[Ni], X[Ni])
                if ALLY[Ni] <= ALLX[Ni] * 1.05 and ALLY[Ni] >= ALLX[Ni] * 0.95:
                    Nopt = Nopt + 1
                    Ntar = Ntar + 1
                    Nthr = Nthr + 1
                elif ALLY[Ni] <= ALLX[Ni] * 1.1 and ALLY[Ni] >= ALLX[Ni] * 0.9:
                    Ntar = Ntar + 1
                    Nthr = Nthr + 1
                elif ALLY[Ni] <= ALLX[Ni] * 1.2 and ALLY[Ni] >= ALLX[Ni] * 0.8:
                    Nthr = Nthr + 1
            URAopt = Nopt / N
            URAtar = Ntar / N
            URAthr = Nthr / N
            URAoptlist.append(URAopt)
            URAtarlist.append(URAtar)
            URAthrlist.append(URAthr)






            ax.text(0.02, 0.98, "(" + chr(96 + subi) + ") N=" + str(N), verticalalignment='top',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize - 5)

            Nlist.append(N)
            ax.text(0.02, 0.51, "R=" + str(r21), verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            Rlist.append(r21)
            ax.text(0.2, 0.09, "P$_\mathrm{T}$", verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            ax.text(0.15, 0.01, str(round(abs(URAthr) * 100, 1)) + "%", verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            # ax.text(0.02, 0.01, "P$_\mathrm{T}$=" + str(round(abs(URAthr) * 100, 1)) + "%", verticalalignment='bottom',
            #         horizontalalignment='left',
            #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            ax.text(0.5, 0.09, "P$_\mathrm{T}$", verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            ax.text(0.45, 0.01, str(round(abs(URAtar) * 100, 1)) + "%", verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            ax.text(0.8, 0.09, "P$_\mathrm{O}$", verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            ax.text(0.75, 0.01, str(round(abs(URAopt) * 100, 1)) + "%", verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            # # ax.text(0.02, 0.33, r'R$^2$=' + str(r21), verticalalignment='bottom', horizontalalignment='left',
            # #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            if SDbias1 > 0:
                ax.text(0.02, 0.59, r'SD=' + str(SDbias1), verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # ax.text(0.02, 0.25, r'RSD=' + str(round(abs(RSDB) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            else:
                ax.text(0.02, 0.59, r'SD=$-$' + str(abs(SDbias1)), verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # ax.text(0.02, 0.25, r'RSD=$-$' + str(round(abs(RSDB) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            #     0.41,0.49
            SDBlist.append(SDbias1)
            if bias1 > 0:
                ax.text(0.02, 0.67, r'Bias=' + str(bias1), verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # ax.text(0.02, 0.41, r'RBias=' + str(round(abs(RMB) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            else:
                ax.text(0.02, 0.67, r'Bias=$-$' + str(abs(bias1)), verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # ax.text(0.02, 0.41, r'RBias=$-$' + str(round(abs(RMB) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            ax.text(0.02, 0.75, r'RMSE=' + str(rmse1), verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            # ax.text(0.02, 0.09, r'RRMSE=' + str(round(rrmse * 100, 1)) + "%", verticalalignment='bottom',
            #         horizontalalignment='left',
            #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            if b1 > 0:
                ax.text(0.02, 0.83, r'y=' + str(a1) + "x+" + str(b1), verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            else:
                ax.text(0.02, 0.83, r'y=' + str(a1) + "x$-$" + str(abs(b1)), verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            # outdf["ID"] = idlist
            outdf["N"] = Nlist
            outdf["R"] = Rlist
            outdf["SDB"] = SDBlist
            outdf["RSDB"] = RSDBlist
            outdf["MB"] = MBlist
            outdf["RMB"] = RMBlist
            outdf["RMSE"] = RMSElist
            outdf["RRMSE"] = RRMSElist
            outdf["URAopt"] = URAoptlist
            outdf["URAtar"] = URAtarlist
            outdf["URAthr"] = URAthrlist

            outdf.to_csv(eachdata + "_" + eachsite + "_result.csv")
            subi = subi + 1
            # 拟合线
            ax.plot(x1, yy1, color='k', linewidth=1.2)
            # y=x线
            # # y=x线
            xxx = [0, 1]
            ax.plot(xxx, xxx, color='black', linewidth=0.8, linestyle='--')

            ax.axis([0, 1.02, 0, 1.02])

            #
            ax.xaxis.set_major_locator(MultipleLocator(0.3))
            ax.yaxis.set_major_locator(MultipleLocator(0.3))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            #
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(fontsize) for label in labels]

            if subi >17:
                ax.set_xlabel("Reference FAPAR", fontdict=font)
            else:
                ax.set_xticklabels([])
            # [2, 7, 12, 17, 22]
            if subi in [2]:
                ax.set_ylabel("MOD FAPAR", fontdict=font)
            elif subi in [6]:
                ax.set_ylabel("MYD FAPAR", fontdict=font)
            elif subi in [10]:
                ax.set_ylabel("VNP FAPAR", fontdict=font)
            elif subi in [14]:
                ax.set_ylabel("GEOV3 FAPAR", fontdict=font)
            elif subi in [18]:
                ax.set_ylabel("GEOV2 FAPAR", fontdict=font)
            else:
                ax.set_yticklabels([])

            if subi in [5, 9, 13, 17, 21]:
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.1)
                # # plt.colorbar(im, cax=cax)
                # cbar = plt.colorbar(sc, cax=cax)
                position = fig.add_axes([0.94, 0.801 - (subi // 4 - 1) * 0.188, 0.01, 0.18])

                # position = fig.add_axes([0.94, 0.613, 0.01, 0.18])
                # position = fig.add_axes([0.94, 0.425, 0.01, 0.18])

                cbar = plt.colorbar(sc, cax=position)

                # cbar.set_label('DOY',fontdict=font)
                # cbar.set_ticks(np.linspace(160, 260, 6))
                cbar.ax.tick_params(labelsize=fontsize)
                labels = cbar.ax.get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]

    # plt.subplots_adjust(left=0.13, right=0.96, bottom=0.08, top=0.98, wspace=0.35, hspace=0.35)
    # plt.subplots_adjust(left=0.06, right=0.96, bottom=0.05, top=0.98, wspace=0.1, hspace=0.1)
    plt.subplots_adjust(left=0.07, right=0.935, bottom=0.05, top=0.98, wspace=0.05, hspace=0.05)

    plt.savefig('Fig9_all.png')  # 保存图片
