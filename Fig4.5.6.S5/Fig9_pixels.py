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
    s = os.sep
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
    subi = 1
    outdf = pd.DataFrame()
    eachdatalist, eachsitelist,Nlist, Rlist, SDBlist, RSDBlist, MBlist, RMBlist, RMSElist, RRMSElist, URAoptlist, URAtarlist, URAthrlist = [], [],[], [], [], [], [], [], [], [], [], [], []

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
                tempTPDdf = TPDdf[(TPDdf[eachdata + "FAPAR" + str(ipixel)] > 0)]
                # tempTPDdf = tempTPDdf[(tempTPDdf[eachdata + "FAPAR" + str(ipixel)] > 0)]

                tempZ = tempTPDdf["doy"].values.tolist()
                tempX = tempTPDdf["HLS" + str(ipixel)].values.tolist()
                tempY = tempTPDdf[eachdata + "FAPAR" + str(ipixel)].values.tolist()
                ALLX.extend(tempX)
                ALLY.extend(tempY)
                ALLDOY.extend(tempZ)

            ax = fig.add_subplot(5, 4, subi)
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
                # if ii > 5:
                Targetupxi.append(ii * 0.01)
                Targetupyi.append(ii * 0.01 * 1.1)
                Targetlowxi.append(ii * 0.01)
                Targetlowyi.append(ii * 0.01 * 0.9)
                # else:
                #     Targetupxi.append(ii * 0.01)
                #     Targetupyi.append(ii * 0.01 + 0.05)
                #     Targetlowxi.append(ii * 0.01)
                #     Targetlowyi.append(ii * 0.01 - 0.05)
            # ax.fill_between(Thresholdlowxi, np.array(Thresholdlowyi), np.array(Thresholdupyi), facecolor="darkgray",
            #                 alpha=1)
            ax.fill_between(Targetlowxi, np.array(Targetlowyi), np.array(Targetupyi), facecolor="darkgray", alpha=1)
            ax.fill_between(Optimallowxi, np.array(Optimallowyi), np.array(Optimalupyi), facecolor="lightgray", alpha=1)

            if subi < 5:
                ax.text(0.02, 1.1, eachsite, verticalalignment='top', horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize)



            sc = ax.scatter(ALLX, ALLY, marker="o", s=20, linewidths=0.2, edgecolors='k', c=ALLDOY, vmin=1, vmax=365,
                            cmap=cm)

            # N = len(ALLY)
            #
            x1 = np.array(ALLX).reshape(-1, 1)
            y1 = np.array(ALLY).reshape(-1, 1)
            regr1 = linear_model.LinearRegression()
            regr1.fit(x1, y1)
            yy1 = regr1.predict(x1)
            r21 = round(np.sqrt(regr1.score(x1, y1)), 2)
            a1 = round(regr1.coef_[0][0], 2)
            b1 = round(regr1.intercept_[0], 2)
            rmse1 = round(math.sqrt(mean_squared_error(x1, y1)), 2)
            bias1 = (round(np.mean(y1 - x1), 2))
            #
            rrmse = round(rmse1 / np.mean(ALLX), 3)
            # PGCOS = str(round(OGCOS * 1.0 / N * 100, 1))
            PGCOS = str(len(ALLX))
            SDbias1 = round(np.std(y1 - x1), 2)
            meanX = np.mean(ALLX)
            RMB = bias1 / meanX
            RRMSE = rmse1 / meanX
            RSDB = SDbias1 / meanX
            N = len(ALLX)
            Nlist.append(N)
            Rlist.append(r21)
            SDBlist.append(SDbias1)
            RSDBlist.append(RSDB)
            MBlist.append(bias1)
            # RMBlist.append(RMB)
            RMSElist.append(rmse1)
            # RRMSElist.append(RRMSE)
            eachdatalist.append(eachdata)
            eachsitelist.append(eachsite)
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
            URAoptlist.append(URAopt*100)
            URAtarlist.append(URAtar*100)
            URAthrlist.append(URAthr*100)

            N = len(ALLX)

            # ax.text(0.02, 0.49, "G=" + PGCOS + "%", verticalalignment='bottom', horizontalalignment='left',
            #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            ax.text(0.02, 0.98, "(" + chr(96 + subi) + ") N="+str(N), verticalalignment='top',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize-5)
            ax.text(0.02, 0.49, "R=" + str(r21), verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            # ax.text(0.02, 0.81, "P$_\mathrm{Thr}$=" + str(round(abs(URAthr) * 100, 1)) + "%", verticalalignment='bottom',
            #         horizontalalignment='left',
            #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            ax.text(0.02, 0.41, "P$_\mathrm{T}$=" + str(round(abs(URAtar) * 100, 1)) + "%", verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            ax.text(0.02, 0.33, "P$_\mathrm{O}$=" + str(round(abs(URAopt) * 100, 1)) + "%", verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            # # ax.text(0.02, 0.33, r'R$^2$=' + str(r21), verticalalignment='bottom', horizontalalignment='left',
            # #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            if SDbias1 > 0:
                ax.text(0.02, 0.57, r'SD=' + str(SDbias1), verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # ax.text(0.02, 0.25, r'RSDB=' + str(round(abs(RSDB) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            else:
                ax.text(0.02, 0.57, r'SD=$-$' + str(abs(SDbias1)), verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # ax.text(0.02, 0.25, r'RSDB=$-$' + str(round(abs(RSDB) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            #     0.41,0.49
            if bias1 > 0:
                ax.text(0.02, 0.65, r'Bias=' + str(bias1), verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # ax.text(0.02, 0.41, r'RMB=' + str(round(abs(RMB) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            else:
                ax.text(0.02, 0.65, r'Bias=$-$' + str(abs(bias1)), verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # ax.text(0.02, 0.41, r'RMB=$-$' + str(round(abs(RMB) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            ax.text(0.02, 0.73, r'RMSE=' + str(rmse1), verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            # ax.text(0.02, 0.09, r'RRMSE=' + str(round(rrmse * 100, 1)) + "%", verticalalignment='bottom',
            #         horizontalalignment='left',
            #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            if b1 > 0:
                ax.text(0.02, 0.81, r'y=' + str(a1) + "x+" + str(b1), verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
            else:
                ax.text(0.02, 0.81, r'y=' + str(a1) + "x$-$" + str(abs(b1)), verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

            subi = subi + 1
            # 拟合线
            ax.plot(x1, yy1, color='k', linewidth=1.2)
            # y=x线
            # # y=x线
            xxx = [0, 1]
            ax.plot(xxx, xxx, color='black', linewidth=0.8, linestyle='--')
            lowxi = []
            lowyi = []
            for ii in range(120):
                if ii > 50:
                    lowxi.append(ii * 0.01)
                    lowyi.append(ii * 0.01 * 0.9)
                else:
                    lowxi.append(ii * 0.01)
                    lowyi.append(ii * 0.01 - 0.05)

            upxi = []
            upyi = []
            for ii in range(120):
                if ii > 50:
                    upxi.append(ii * 0.01)
                    upyi.append(ii * 0.01 * 1.1)
                else:
                    upxi.append(ii * 0.01)
                    upyi.append(ii * 0.01 + 0.05)

            # ax.fill_between(lowxi, np.array(lowyi), np.array(upyi), alpha=0.2)
            ax.axis([-0.29, 1.05, -0.29, 1.05])
            ax.axis([0.05, 1, 0.05, 1])

            # ax.axis([-0.29, 0.5, 0, 0.2])
            # #
            ax.xaxis.set_major_locator(MultipleLocator(0.3))
            ax.yaxis.set_major_locator(MultipleLocator(0.3))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            #
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(fontsize) for label in labels]

            if subi > 17:
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
    plt.subplots_adjust(left=0.07, right=0.95, bottom=0.05, top=0.98, wspace=0.05, hspace=0.05)

    plt.savefig('Fig9_pixels.png')  # 保存图片
    # outdf["ID"] = idlist
    outdf["N"] = Nlist
    outdf["R"] = Rlist
    outdf["RMSE"] = RMSElist
    outdf["Bias"] = MBlist
    outdf["SDB"] = SDBlist

    outdf["URAopt"] = URAoptlist
    outdf["URAtar"] = URAtarlist
    outdf["URAthr"] = URAthrlist

    outdf["site"] = eachsitelist
    outdf["data"] = eachdatalist
    outdf.to_csv("pixels_result.csv")
