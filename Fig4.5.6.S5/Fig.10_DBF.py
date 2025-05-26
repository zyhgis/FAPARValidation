# -*- coding: utf-8 -*-

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
from scipy.stats import pearsonr
import pandas as pd
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

    fontsize = 24
    font = {'family': 'Times New Roman',
            'color': 'k',
            'weight': 'normal',
            'size': fontsize-2}
    legendfont = {'family': 'Times New Roman',
                  'size': fontsize}
    fig = plt.figure(figsize=(8, 20), dpi=300)

    Site = ["Deciduous Forests"]
    Products = ["MOD", "MYD", "VNP", "PROBAV1000", "PROBAV300"]
    ProductsLabel = {"MOD": "MOD", "MYD": "MYD", "VNP": "VNP", "P300": "GEOV3", "P1000": "GEOV2"}
    QAdict = {"MOD": ["RT", "VI"],
              "MYD": ["RT", "VI"],
              "VNP": ["RT", "VI"],
              "PROBAV1000": ["PF", "LF"],
              "PROBAV300": ["DR", "CR"]}

    for eachSite in Site:
        outdf = pd.DataFrame()
        idlist, Nlist, Rlist, SDBlist, RSDBlist, MBlist, RMBlist, RMSElist, RRMSElist, URAoptlist, URAtarlist, URAthrlist = [], [], [], [], [], [], [], [], [], [], [], []

        i = 1
        subi=1
        for eachProduct in Products:
            QAlist = QAdict[eachProduct]
            if eachProduct in ["MOD","MYD","VNP"]:
                TPDdf = pd.read_csv(r".\Data\CA-TPD_HLS_MODIS_FAPAR_RMSE_QC.csv")
                ncols = TPDdf.shape[1]
                TPDnpixels = (ncols - 3) // 10

                Bardf = pd.read_csv(r".\Data\US-Bar_HLS_MODIS_FAPAR_RMSE_QC.csv")
                ncols = Bardf.shape[1]
                Barnpixels = (ncols - 3) // 10

                HFdf = pd.read_csv(r".\Data\US-HF_HLS_MODIS_FAPAR_RMSE_QC.csv")
                ncols = HFdf.shape[1]
                HFnpixels = (ncols - 3) // 10

            else:
                TPDdf = pd.read_csv(r".\Data\CA-TPD_HLS_"+eachProduct+"_FAPAR_RMSE_QC.csv")
                ncols = TPDdf.shape[1]
                TPDnpixels = (ncols - 3) // 4

                Bardf = pd.read_csv(r".\Data\US-Bar_HLS_" + eachProduct + "_FAPAR_RMSE_QC.csv")
                ncols = Bardf.shape[1]
                Barnpixels = (ncols - 3) // 4

                HFdf = pd.read_csv(r".\Data\US-HF_HLS_" + eachProduct + "_FAPAR_RMSE_QC.csv")
                ncols = HFdf.shape[1]
                HFnpixels = (ncols - 3) // 4

            QAi=0
            for eachQA in QAlist:
                ALLX = []
                ALLY = []
                ALLZ = []
                QAi=QAi+1
                if QAi==1:
                    marker1=1
                    marker2=2
                else:
                    marker1 = 3
                    marker2 = 4
                for ipixel in range(1, TPDnpixels + 1):
                    tempTPDdf = TPDdf[((TPDdf[eachProduct + "QA" + str(ipixel)] == marker1) | (TPDdf[eachProduct + "QA" + str(ipixel)] == marker2)) & (TPDdf[eachProduct + "FAPAR" + str(ipixel)] >0)]
                    tempZ = tempTPDdf["doy"].values.tolist()
                    tempX = tempTPDdf["HLS" + str(ipixel)].values.tolist()
                    tempY = tempTPDdf[eachProduct + "FAPAR" + str(ipixel)].values.tolist()
                    ALLX.extend(tempX)
                    ALLY.extend(tempY)
                    ALLZ.extend(tempZ)

                for ipixel in range(1, Barnpixels + 1):
                    tempBardf = Bardf[((Bardf[eachProduct + "QA" + str(ipixel)] == marker1) | (Bardf[eachProduct + "QA" + str(ipixel)] == marker2)) & (Bardf[eachProduct + "FAPAR" + str(ipixel)] >0)]
                    tempZ = tempBardf["doy"].values.tolist()
                    tempX = tempBardf["HLS" + str(ipixel)].values.tolist()
                    tempY = tempBardf[eachProduct + "FAPAR" + str(ipixel)].values.tolist()
                    ALLX.extend(tempX)
                    ALLY.extend(tempY)
                    ALLZ.extend(tempZ)

                for ipixel in range(1, HFnpixels + 1):
                    tempHFdf = HFdf[((HFdf[eachProduct + "QA" + str(ipixel)] == marker1) | (HFdf[eachProduct + "QA" + str(ipixel)] == marker2)) & (HFdf[eachProduct + "FAPAR" + str(ipixel)] >0)]
                    tempZ = tempHFdf["doy"].values.tolist()
                    tempX = tempHFdf["HLS" + str(ipixel)].values.tolist()
                    tempY = tempHFdf[eachProduct + "FAPAR" + str(ipixel)].values.tolist()
                    ALLX.extend(tempX)
                    ALLY.extend(tempY)
                    ALLZ.extend(tempZ)

                print(i)
                ax = fig.add_subplot(5, 2, i)
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

                    Targetupxi.append(ii * 0.01)
                    Targetupyi.append(ii * 0.01 * 1.1)
                    Targetlowxi.append(ii * 0.01)
                    Targetlowyi.append(ii * 0.01 * 0.9)

                    # if ii > 50:
                    #     Targetupxi.append(ii * 0.01)
                    #     Targetupyi.append(ii * 0.01 * 1.1)
                    #     Targetlowxi.append(ii * 0.01)
                    #     Targetlowyi.append(ii * 0.01 * 0.9)
                    # else:
                    #     Targetupxi.append(ii * 0.01)
                    #     Targetupyi.append(ii * 0.01 + 0.05)
                    #     Targetlowxi.append(ii * 0.01)
                    #     Targetlowyi.append(ii * 0.01 - 0.05)
                if subi<11:
                    # ax.fill_between(Thresholdlowxi, np.array(Thresholdlowyi), np.array(Thresholdupyi), facecolor="dimgray",
                    #                 alpha=1)
                    ax.fill_between(Targetlowxi, np.array(Targetlowyi), np.array(Targetupyi), facecolor="darkgray", alpha=1)
                    ax.fill_between(Optimallowxi, np.array(Optimallowyi), np.array(Optimalupyi), facecolor="lightgray",
                                    alpha=1)
                # if i ==16:
                #     continue
                if i == 1:
                    ax.text(0.02, 1.1, "(II) "+eachSite, verticalalignment='top', horizontalalignment='left',
                            transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize)

                if i in [1, 3, 5, 7, 9]:
                    if eachProduct=="PROBAV1000":
                        ax.set_ylabel("GEOV2", fontdict=font)
                    elif eachProduct=="PROBAV300":
                        ax.set_ylabel("GEOV3", fontdict=font)
                    else:
                        ax.set_ylabel(eachProduct, fontdict=font)
                else:
                    ax.set_yticklabels([])

                if i in [9,10]:
                    ax.set_xlabel("Reference FAPAR", fontdict=font)
                else:
                    ax.set_xticklabels([])
                sc = ax.scatter(ALLX, ALLY, marker="o", s=20, linewidths=0.2, edgecolors='k', c=ALLZ, vmin=1,
                                vmax=365,
                                cmap=cm)

                if len(ALLX) == 0:
                    ax.axis('off')
                    i=i+1
                    continue
                if i == 16:
                    ax.axis('off')
                    i=i+1
                    continue
                # ax.text(0.02, 0.98, "(" + chr(96 + subi) + ") "+eachQA, verticalalignment='top', horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize)
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
                rrmse = round(rmse1 / np.mean(ALLX), 3)

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
                RMBlist.append(RMB)
                RMSElist.append(rmse1)
                RRMSElist.append(RRMSE)
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


                N = len(ALLX)

                # ax.text(0.02, 0.4, eachQA, verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize - 5)

                # ax.text(0.02, 0.49, "G=" + PGCOS + "%", verticalalignment='bottom', horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                ax.text(0.02, 0.98, "(" + chr(96 + subi) + ") " + eachQA, verticalalignment='top',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize - 5)
                ax.text(0.02, 0.90, "N=" + str(N), verticalalignment='top',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize - 5)
                ax.text(0.02, 0.43, "R=" + str(r21), verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

                ax.text(0.02, 0.35, "P$_\mathrm{T}$=" + str(round(abs(URAtar) * 100, 1)) + "%",
                        verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                ax.text(0.02, 0.27, "P$_\mathrm{O}$=" + str(round(abs(URAopt) * 100, 1)) + "%",
                        verticalalignment='bottom',
                        horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

                # ax.text(0.2, 0.09, "P$_\mathrm{T}$", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                #
                # ax.text(0.15, 0.01, str(round(abs(URAthr) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                #
                # # ax.text(0.02, 0.01, "P$_\mathrm{T}$=" + str(round(abs(URAthr) * 100, 1)) + "%", verticalalignment='bottom',
                # #         horizontalalignment='left',
                # #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                #
                # ax.text(0.5, 0.09, "P$_\mathrm{T}$", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                #
                # ax.text(0.45, 0.01, str(round(abs(URAtar) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                #
                # ax.text(0.8, 0.09, "P$_\mathrm{O}$", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # ax.text(0.75, 0.01, str(round(abs(URAopt) * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

                # # ax.text(0.02, 0.33, r'R$^2$=' + str(r21), verticalalignment='bottom', horizontalalignment='left',
                # #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

                if SDbias1 > 0:
                    ax.text(0.02, 0.51, r'SD=' + str(SDbias1), verticalalignment='bottom', horizontalalignment='left',
                            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                    # ax.text(0.02, 0.25, r'RSD=' + str(round(abs(RSDB) * 100, 1)) + "%", verticalalignment='bottom',
                    #         horizontalalignment='left',
                    #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                else:
                    ax.text(0.02, 0.51, r'SD=$-$' + str(abs(SDbias1)), verticalalignment='bottom',
                            horizontalalignment='left',
                            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                    # ax.text(0.02, 0.25, r'RSD=$-$' + str(round(abs(RSDB) * 100, 1)) + "%", verticalalignment='bottom',
                    #         horizontalalignment='left',
                    #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                #     0.41,0.49
                if bias1 > 0:
                    ax.text(0.02, 0.59, r'Bias=' + str(bias1), verticalalignment='bottom', horizontalalignment='left',
                            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                    # ax.text(0.02, 0.41, r'RBias=' + str(round(abs(RMB) * 100, 1)) + "%", verticalalignment='bottom',
                    #         horizontalalignment='left',
                    #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                else:
                    ax.text(0.02, 0.59, r'Bias=$-$' + str(abs(bias1)), verticalalignment='bottom',
                            horizontalalignment='left',
                            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                    # ax.text(0.02, 0.41, r'RBias=$-$' + str(round(abs(RMB) * 100, 1)) + "%", verticalalignment='bottom',
                    #         horizontalalignment='left',
                    #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

                ax.text(0.02, 0.67, r'RMSE=' + str(rmse1), verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

                # ax.text(0.02, 0.09, r'RRMSE=' + str(round(rrmse * 100, 1)) + "%", verticalalignment='bottom',
                #         horizontalalignment='left',
                #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)

                if b1 > 0:
                    ax.text(0.02, 0.75, r'y=' + str(a1) + "x+" + str(b1), verticalalignment='bottom',
                            horizontalalignment='left',
                            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                else:
                    ax.text(0.02, 0.75, r'y=' + str(a1) + "x$-$" + str(abs(b1)), verticalalignment='bottom',
                            horizontalalignment='left',
                            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 5)
                # 拟合线
                ax.plot(x1, yy1, color='k', linewidth=1.2)
                # y=x线
                # # y=x线
                xxx = [0, 1]
                ax.plot(xxx, xxx, color='black', linewidth=0.8, linestyle='--')

                ax.axis([0.05, 1.02, 0.05, 1.02])
                #
                ax.xaxis.set_major_locator(MultipleLocator(0.3))
                ax.yaxis.set_major_locator(MultipleLocator(0.3))
                ax.xaxis.set_minor_locator(MultipleLocator(0.1))
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                #
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(fontsize) for label in labels]
                # if i in [9]:
                #     position = fig.add_axes([0.555, 0.04, 0.02, 0.17])
                #     cbar = plt.colorbar(sc, cax=position)
                #     labels = cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels()
                #     [label.set_fontname('Times New Roman') for label in labels]
                #     [label.set_fontsize(fontsize) for label in labels]
                if i in [10]:
                    position = fig.add_axes([0.8, 0.12, 0.02, 0.1])
                    cbar = plt.colorbar(sc, cax=position)
                    labels = cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels()
                    [label.set_fontname('Times New Roman') for label in labels]
                    [label.set_fontsize(fontsize) for label in labels]
                i = i + 1
                subi=subi+1

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

        outdf.to_csv(eachSite + "_result.csv")
        plt.subplots_adjust(left=0.12, right=0.99, bottom=0.04, top=0.98, wspace=0.05, hspace=0.05)

        plt.savefig('Fig.HLS_FAPAR_QA_DBF.png')  # 保存图片
