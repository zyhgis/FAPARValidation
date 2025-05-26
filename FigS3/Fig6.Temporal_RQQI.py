# -*- coding: utf-8 -*-
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
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
    s = os.sep

    fig = plt.figure(figsize=(24, 16), dpi=300)
    fontsize = 20
    font = {'family': 'Times New Roman',
            'color': 'k',
            'weight': 'normal',
            'size': fontsize + 18, }
    legendfont = {'family': 'Times New Roman',
                  'size': fontsize + 18}


    sitelist = ["CA-TP4", "CA-TPD", "US-Bar", "US-HF"]
    subi=0

    for eachsite in sitelist:
        subi=subi+1

        ax = fig.add_subplot(4,1,subi)

        MODdf = pd.read_csv("./Data" + s + eachsite + "_MODFAPAR.csv")
        MODdf = MODdf[MODdf["year"] > 2009]
        MODdf["X"] = MODdf['year'] * 1000 + MODdf['doy'] * 2.75
        MODdf['RQQI']=MODdf["FAPARstd"]/MODdf["FAPAR"]*100
        PEMODDOY = MODdf["X"].values.tolist()
        PEMODfaparstd = MODdf["RQQI"].values.tolist()

        MYDdf = pd.read_csv("./Data" + s + eachsite + "_MYDFAPAR.csv")
        MYDdf = MYDdf[MYDdf["year"] > 2009]
        MYDdf["X"] = MYDdf['year'] * 1000 + MYDdf['doy'] * 2.75
        MYDdf['RQQI']=MYDdf["FAPARstd"]/MYDdf["FAPAR"]*100
        PEMYDDOY = MYDdf["X"].values.tolist()
        PEMYDfaparstd = MYDdf["RQQI"].values.tolist()

        VNPdf = pd.read_csv("./Data" + s + eachsite + "_VNPFAPAR.csv")
        VNPdf = VNPdf[VNPdf["year"] > 2009]
        VNPdf["X"] = VNPdf['year'] * 1000 + VNPdf['doy'] * 2.75
        VNPdf['RQQI']=VNPdf["FAPARstd"]/VNPdf["FAPAR"]*100
        PEVNPDOY = VNPdf["X"].values.tolist()
        PEVNPfaparstd = VNPdf["RQQI"].values.tolist()

        PROBAV300df = pd.read_csv("./Data" + s + eachsite + "_PROBAV300FAPAR.csv")
        PROBAV300df = PROBAV300df[PROBAV300df["year"] > 2009]
        PROBAV300df["X"] = PROBAV300df['year'] * 1000 + PROBAV300df['doy'] * 2.75
        PROBAV300df['RQQI']=PROBAV300df["FAPARstd"]/PROBAV300df["FAPAR"]*100

        PEPROBAV300DOY = PROBAV300df["X"].values.tolist()
        PEPROBAV300faparstd = PROBAV300df["RQQI"].values.tolist()

        PROBAV1000df = pd.read_csv("./Data" + s + eachsite + "_PROBAV1000FAPAR.csv")
        PROBAV1000df = PROBAV1000df[PROBAV1000df["year"] > 2009]
        PROBAV1000df["X"] = PROBAV1000df['year'] * 1000 + PROBAV1000df['doy'] * 2.75
        PROBAV1000df['RQQI']=PROBAV1000df["FAPARstd"]/PROBAV1000df["FAPAR"]*100
        PEPROBAV1000DOY = PROBAV1000df["X"].values.tolist()
        PEPROBAV1000faparstd = PROBAV1000df["RQQI"].values.tolist()

        # if EPSQQIDOY
        ax.plot(PEMODDOY, PEMODfaparstd, c='blue', linewidth=2, label='MOD')
        ax.plot(PEMYDDOY, PEMYDfaparstd, c='r', linewidth=2, label='MYD')
        ax.plot(PEVNPDOY, PEVNPfaparstd, c='m', linewidth=2, label='VNP')

        ax.plot(PEPROBAV1000DOY, PEPROBAV1000faparstd, c='lawngreen', linewidth=2, label="GEOV2")
        ax.plot(PEPROBAV300DOY, PEPROBAV300faparstd, c='green', linewidth=2, label="GEOV3")

        ax.text(0.005, 0.97, "(" + chr(96 + subi+5) + ") "+eachsite, verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, fontname="Times New Roman", fontsize=fontsize + 12)
        # ax.text(0.03, 0.9, "(" + chr(96 + 3) + ")", verticalalignment='bottom', horizontalalignment='left',
        #         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize)
        if subi == 1:
            ax.legend(loc='lower left', bbox_to_anchor=(0.42, 0.65), ncol=5, frameon=False, labelspacing=0.1,
                      handlelength=1, handletextpad=0.2, columnspacing=1, prop=legendfont)

        xtic = []
        for i in range(2013, 2018):
            xtic.append(float(str(i) + '001'))
            plt.xticks(xtic, fontsize=fontsize+12)
        # ax.axis([2013990, 2021010, -5, 70])
        ax.axis([2012890, 2018610, -10, 70])

        # ax.set_xlim(2009600, 2012000)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        # ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.xaxis.set_minor_locator(MultipleLocator(200))
        if subi==3:
            ax.set_ylabel("Relative QQIs (%)", fontdict=font,y=1.1)
        # if subi==1:
        ax.set_xticks([2013001,2014001, 2015001, 2016001, 2017001, 2018001])
        ax.get_xaxis().get_major_formatter().set_scientific(False)

        if subi ==4:
            ax.set_xlabel("Day of year", fontdict=font)

        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(fontsize + 12) for label in labels]


    # plt.subplots_adjust(left=0.045, right=0.99, bottom=0.05, top=0.98,wspace=0.15, hspace=0.15)
    plt.subplots_adjust(left=0.055, right=0.99, bottom=0.07, top=0.98,wspace=0.2, hspace=0.2)

    plt.savefig('Fig6.Temporal_RQQI2.png')  # 保存图片
