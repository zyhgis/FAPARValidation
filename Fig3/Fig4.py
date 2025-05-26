# -*- coding: utf-8 -*-
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
from matplotlib.pyplot import *

def listdir(path):
    list_name=[]
    for file in os.listdir(path):
        # print file
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    return list_name
def nasslistdir(path):
    list_name=[]
    for file in os.listdir(path):
        # print file
        # clipprojectMOD15A2HA2010001
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, file[0:27])
            list_name.append(file_path)
    list_name = sorted(list(set(list_name)))
    return list_name

def vnplistdir(path):
    list_name=[]
    for file in os.listdir(path):
        # print file
        # clipprojectMOD15A2HA2010001
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, file[0:28])
            list_name.append(file_path)
    list_name = sorted(list(set(list_name)))
    return list_name

def geov2rmsemeandata(tifdata):
    sumdata=0
    pixelcount=0.0
    for i in tifdata:
        for j in i:
            if j<=235:
                sumdata=sumdata+j
                pixelcount=pixelcount+1
    if pixelcount==0:
        return -1
    else:
        meanvalue=sumdata/pixelcount
        return meanvalue

def geov2faparmeandata(tifdata):
    sumdata=0
    pixelcount=0.0
    for i in tifdata:
        for j in i:
            if j<=235:
                sumdata=sumdata+j
                pixelcount=pixelcount+1
    if pixelcount==0:
        return -1
    else:
        meanvalue=sumdata/pixelcount
        return meanvalue

def nassmeandata(tifdata):
    sumdata=0
    pixelcount=0.0
    for i in tifdata:
        for j in i:
            if j<=100:
                sumdata=sumdata+j
                pixelcount=pixelcount+1
    if pixelcount==0:
        return -1
    else:
        meanvalue=sumdata/pixelcount
        return meanvalue

if __name__ == "__main__":
    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    s = os.sep

    fig = plt.figure(figsize=(24, 16), dpi=300)
    fontsize = 12
    font = {'family': 'Times New Roman',
            'color': 'k',
            'weight': 'normal',
            'size': fontsize + 18, }
    legendfont = {'family': 'Times New Roman',
                  'size': fontsize + 18}

    # ax.plot(GeoV2QQIDOY, GeoV2QQI, c='k', linewidth=2, s=6,label='GeoV2')
    GEOV2DATA = {}
    GEOV2DOY = []
    GEOV2RMSE = []
    GEOV2fapar = []


    GEOV3DATA = {}
    GEOV3DOY = []
    GEOV3RMSE = []
    GEOV3fapar = []

    MODDATA = {}
    MODDOY = []
    MODRMSE = []
    MODfapar = []


    MYDDATA = {}
    MYDDOY = []
    MYDRMSE = []
    MYDfapar = []


    VNPDATA = {}
    VNPDOY = []
    VNPRMSE = []
    VNPfapar = []


    PEGEOV2DATA = {}
    PEGEOV2DOY = []
    PEGEOV2RMSE = []
    PEGEOV2fapar = []

    PEGEOV3DATA = {}
    PEGEOV3DOY = []
    PEGEOV3RMSE = []
    PEGEOV3fapar = []

    PEMODDATA = {}
    PEMODDOY = []
    PEMODRMSE = []
    PEMODfapar = []

    PEMYDDATA = {}
    PEMYDDOY = []
    PEMYDRMSE = []
    PEMYDfapar = []

    PEVNPDATA = {}
    PEVNPDOY = []
    PEVNPRMSE = []
    PEVNPfapar = []
    sitelist = ["CA-TP4", "CA-TPD", "US-Bar", "US-HF"]
    subi=0


    for eachsite in sitelist:
        subi=subi+1

        ax = fig.add_subplot(4,1,subi)
        HLSdf = pd.read_csv("./Data"+s+eachsite+"_HLS_InsFAPAR.csv")
        df=pd.read_csv("./Data"+s+eachsite+"_Field_InsFAPAR.csv")
        df["X"]=df["Year"]*1000+df["DOY"]*2.75
        X=df["X"]
        Y=df["FieldFAPAR"]
        ax.scatter(X, Y, edgecolors='r',c="none", marker="o", s=200,linewidths=1, label=r"${in \ situ}$ FAPAR")

        HLSdf["X"] = HLSdf["year"] * 1000 + HLSdf["doy"] * 2.75
        L30df=HLSdf[HLSdf["sensor"]=="L30"]
        L30HLSX = L30df["X"]
        L30HLSY = L30df["HLSFAPAR"]
        ax.scatter(L30HLSX, L30HLSY, edgecolors='g',c="none", marker="o", s=200,linewidths=3, label=r"HLS FAPAR (Landsat 8)")

        S30df = HLSdf[HLSdf["sensor"] == "S30"]
        S30HLSX = S30df["X"]
        S30HLSY = S30df["HLSFAPAR"]
        ax.scatter(S30HLSX, S30HLSY, edgecolors='b', c="none", marker="d", s=200, linewidths=3, label=r"HLS FAPAR (Sentinel 2)")

        ax.text(0.005, 0.13, "(" + chr(96 + subi) + ") "+eachsite, verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize + 12)
        # ax.text(0.03, 0.9, "(" + chr(96 + 3) + ")", verticalalignment='bottom', horizontalalignment='left',
        #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize)
        if subi == 1:
            ax.legend(loc='lower left', bbox_to_anchor=(0.76, 0.001), ncol=1, frameon=False, labelspacing=0.5,
                      handlelength=1,
                      prop=legendfont)
        # xtic = []
        # for i in range(2012, 2021):
        #     xtic.append(float(str(i) + '001'))
        #     plt.xticks(xtic, fontsize=fontsize+12)
        # if subi in [1,2,3,4,5]:
        #     ax.axis([2011890, 2018100, -0.08, 1.1])
        #     ax.set_xticks([2012001,2013001, 2014001, 2015001, 2016001, 2017001, 2018001])
        #
        if subi in [1,2,3,4,5]:
            ax.axis([2012890, 2018100, -0.08, 1.1])
            ax.set_xticks([2013001,2013001, 2014001, 2015001, 2016001, 2017001, 2018001])

        # if subi in [4]:
        #     ax.axis([2011890, 2015800, -0.08, 1.1])
        #     ax.set_xticks([2012001, 2013001, 2014001, 2015001])
        # if subi in [5]:
        #     ax.axis([2011890, 2021150, -0.08, 1.1])
        #     ax.set_xticks([2012001, 2013001, 2014001, 2015001, 2016001, 2017001, 2018001, 2019001, 2020001, 2021001])
        #
        ax.yaxis.set_major_locator(MultipleLocator(0.3))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(200))
        ax.set_ylabel("FAPAR", fontdict=font)

        ax.get_xaxis().get_major_formatter().set_scientific(False)

        if subi ==4:
            ax.set_xlabel("Day of year", fontdict=font)

        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(fontsize + 12) for label in labels]


    plt.subplots_adjust(left=0.045, right=0.99, bottom=0.06, top=0.98,wspace=0.15, hspace=0.15)

    plt.savefig('Fig4.Temporal_Field_HLS_FAPAR4.png')  # 保存图片
