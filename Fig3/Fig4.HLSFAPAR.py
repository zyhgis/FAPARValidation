#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/20 20:17
# @Author  : Yinghui Zhang
# @File    : Extract_Point_GDAL.py
# @Software: PyCharm
import numpy as np
from osgeo import gdal
from osgeo import osr
import os
import pandas as pd
import datetime
# -*- coding: utf-8 -*-
import  pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import math
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.pyplot import *
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'STIXGeneral:italic'
# rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
def get_file_info(in_file_path):
    '''
    根据指定的图像文件路径，以只读的方法打开图像
    :param in_file_path: 输入文件路径
    :return: gdal数据集、地理空间坐标系、投影坐标系、栅格影像的大小相关信息
    '''
    #
    pcs=None
    gcs=None
    shape=None
    #
    if in_file_path.endswith(".tif") or in_file_path.endswith(".TIF"):
        dataset=gdal.Open(in_file_path)
        pcs=osr.SpatialReference()
        pcs.ImportFromWkt(dataset.GetProjection())
        #
        gcs=pcs.CloneGeogCS()
        #
        extend=dataset.GetGeoTransform()
        #
        shape=(dataset.RasterXSize,dataset.RasterYSize)
    else:
        raise("Unsupported file format!")

    return dataset,gcs,pcs,extend,shape
def lonlat_to_xy(gcs,pcs,lon,lat):
    '''
    经纬度坐标转换为投影坐标
    :param gcs: 地理空间坐标信息，可由get_file_info()函数获取
    :param pcs: 投影坐标信息，可由get_file_info()函数获取
    :param lon: 经度坐标
    :param lat: 纬度坐标
    :return: 地理空间坐标对应的投影坐标
    '''
    #
    ct=osr.CoordinateTransformation(gcs,pcs)
    coordinates=ct.TransformPoint(lon,lat)
    #
    return coordinates[0],coordinates[1],coordinates[2]

def xy_to_lonlat(gcs,pcs,x,y):
    '''
    投影坐标转换为经纬度坐标
    :param gcs: 地理空间坐标信息，可由get_file_info()函数获取
    :param pcs: 投影坐标信息，可由get_file_info()函数获取
    :param x: 像元的行号
    :param y: 像元的列号
    :return: 投影坐标对应的地理空间坐标
    '''
    #
    ct=osr.CoordinateTransformation(pcs,gcs)
    lon,lat,_=ct.TransformPoint(x,y)
    #
    return lon,lat

def xy_to_rowcol(extend,x,y):
    '''
    根据GDAL的六参数模型将给定的投影坐标系转为影像图上坐标（行列号）
    :param extend: 图像的空间范围
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标（x,y）对应的影像图上行列号（row,col）
    '''
    #
    a=np.array([[extend[1],extend[2]],[extend[4],extend[5]]])
    b=np.array([x-extend[0],y-extend[3]])
    #
    row_col=np.linalg.solve(a,b)
    row=int(np.floor(row_col[1]))
    col=int(np.floor(row_col[0]))
    #
    return row,col

def rowcol_to_xy(extend,row,col):
    '''
    图像坐标转换为投影坐标
    根据GDAL的六参数模型将影像上坐标（行列号）转为投影坐标或地理坐标
    :param extend: 图像的空间范围
    :param row: 像元的行号
    :param col: 像元的列号
    :return: 行列号（row,col)对应的投影坐标（x，y）
    '''
    #
    x=extend[0]+row*extend[1]+col*extend[2]
    y=extend[3]+row*extend[4]+col*extend[5]
    #
    return x,y

def get_value_by_coordinates(img,gcs,pcs,extend,shape,coordinates,coordinates_type="rowcol"):
    '''
    直接根据图像坐标，或者依据GDAL的六参数模型将给定的投影、地理坐标转为影像图上坐标后，返回对应影像像元值

    :param file_path:图像文件的路径
    :param coordinates: 坐标，2个元素的元组
    :param coordinates_type: "rowcol","xy","lonlat"
    :return: 指定坐标的像元值
    '''
    #
    # img=dataset.GetRasterBand(1).ReadAsArray()
    value=[]
    #
    if coordinates_type=="rowcol":
        # eachvalue=img[coordinates[0],coordinates[1]]
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                eachvalue = img[coordinates[0]+i, coordinates[1]+j]

                value.append(eachvalue)

    elif coordinates_type=="lonlat":
        x,y,_=lonlat_to_xy(gcs,pcs,coordinates[0],coordinates[1])
        row,col=xy_to_rowcol(extend,x,y)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                eachvalue = img[row+i, col+j]

                value.append(eachvalue)
        # value=img[row,col]
    elif coordinates_type=="xy":
        row,col=xy_to_rowcol(extend,coordinates[0],coordinates[1])
        # value=img[row,col]
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                eachvalue = img[row+i, col+j]
                value.append(eachvalue)
    else:
        raise('''"coordinates_type":Wrong parameters input''')
    #
    return value


def listdir(path):
    list_name=[]
    for file in os.listdir(path):
        # print file
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    return list_name
def TIFlistdir(path,tile):
    list_name = []
    for file in os.listdir(path):
        # print(os.path.splitext(file)[0][-11:])
        # G:\HLS\L8\Fmask\HLS.L30.T31TEJ.2013106T103128.v2.0.Fmask.tif
        filesplit=file.split(".")
        if filesplit[-1]=="tif" and filesplit[2]==tile:
            file=".".join(filesplit[:-2])
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    list_name=sorted(list(set(list_name)))
    return list_name
if __name__ == '__main__':
    s=os.sep
    sitelist = [ "US-Bar", "US-HF", "US-Prr", "US-Uaf"]
    tilesdict = {"CA-TP4":["T17TNH"], "CA-TPD":["T17TNH"],"US-Bar": ["T18TYP", "T19TCJ"], "US-HF": ["T18TYN"],
                 "US-Prr": ["T06WVT"], "US-Uaf": ["T06WVS", "T06WVT"]}
    latdict = {"CA-TP4":42.7102, "CA-TPD":42.6353,"US-Bar": 44.0646, "US-HF": 42.5353,
                 "US-Prr": 65.1237, "US-Uaf": 64.8663}
    londict = {"CA-TP4":-80.3574, "CA-TPD":-80.5577,"US-Bar": -71.2881, "US-HF": -72.1899,
                 "US-Prr": -147.4876, "US-Uaf": -147.8555}

    sitelist = ["CA-TP4", "CA-TPD", "US-Bar", "US-HF"]

    filepath="D:\BaiduSyncdisk\Code\FAPAR_Validation_NA\HLS_Clear_Data\L30_Clear"
    basepath="E:\HLS\Clip2"
    FAPARpath=r"D:\FAPAR_Validation_NA\2024\FAPAR\Result2"
    # L30Band = ["B03", "B04", "B05", "B06", "SZA", "SAA", "VAA", "VZA", "Fmask"]
    fieldPATH="D:\FAPAR_Validation_NA\RetrieveFAPAR\ClearValidationData"


    cdict = []

    for x in range(0, 250):
        hsv = ((250 - x) / 360.0, 1, 1)
        rgb = colors.hsv_to_rgb(hsv)
        cdict.append(rgb)
        # ax.scatter(100,x,color=(r,0.35,b),marker="s")
    cm = colors.ListedColormap(cdict, 'zyh')

    fontsize = 20
    font = {'family': 'Times New Roman',
            'color': 'k',
            'weight': 'normal',
            'size': fontsize + 2,
            }
    legendfont = {'family': 'Times New Roman',
                  'size': fontsize}
    makersize = 100


    for eachsite in sitelist:
        # for i in range(9):
        df = pd.DataFrame()
        yearlist = []
        DOYlist = []
        tilelist = []
        sensorlist = []
        redbandlist = []
        nirbandlist = []
        swirbandlist = []
        fmaskbandlist = []
        szabandlist = []
        fieldFAPARlist = []
        siteoutlist = []
        tiles = tilesdict[eachsite]
        lat=latdict[eachsite]
        lon=londict[eachsite]
        coordinates=[lon,lat]

        # E:\HLS\Clip\CA-TP4\L30
        for eachtile in tiles:
            filepath = FAPARpath + s + eachsite
            Tiflist = TIFlistdir(filepath,eachtile)
            for eachfile in Tiflist:
                # HLS.S30.T17TNH.2020342T161649.v2.0.FAPAR.tif
                tempyear=int(os.path.split(eachfile)[1][15:19])
                tempdoy=int(os.path.split(eachfile)[1][19:22])
                eachsensor = os.path.split(eachfile)[1][4:7]
                if eachsensor == "L30":
                    bandfile = Tiflist[0] + ".FAPAR.tif"
                    dataset, gcs, pcs, extend, shape = get_file_info(bandfile)
                    x, y, _ = lonlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
                    row, col = xy_to_rowcol(extend, x, y)

                filesplit=eachfile.split(".")

                bandi=0
                try:
                    # "B04", "B05", "B06",
                    bandfile = eachfile + ".FAPAR.tif"
                    dataset, gcs, pcs, extend, shape = get_file_info(bandfile)
                    fapardata = dataset.GetRasterBand(1).ReadAsArray()
                    tempfapar_array = fapardata[row-1:row+2, col-1:col+2]
                    faparvalue=np.mean(tempfapar_array[tempfapar_array>0])
                    faparvaluelen=len(tempfapar_array[tempfapar_array>0])
                    if faparvaluelen==0:
                        faparvalue=0
                    if faparvalue==0:
                        continue
                    bandfile = eachfile + ".FAPARstd.tif"
                    dataset, gcs, pcs, extend, shape = get_file_info(bandfile)
                    faparstddata = dataset.GetRasterBand(1).ReadAsArray()
                    tempfaparstd_array = faparstddata[row - 1:row + 2, col - 1:col + 2]
                    faparstdvalue = np.mean(tempfaparstd_array[tempfaparstd_array > 0])
                    # faparsvaluelen = len(tempfaparstd_array[tempfaparstd_array > 0])
                    if faparvaluelen == 0:
                        faparstdvalue = 0

                    # bandfile = basepath+ s + eachsite + s + eachsensor+s+os.path.split(eachfile)[1] + ".Fmask.tif"
                    # dataset, gcs, pcs, extend, shape = get_file_info(bandfile)
                    # fmaskbanddata = dataset.GetRasterBand(1).ReadAsArray()
                    # fmaskvalue = fmaskbanddata[row, col]

                    bandfile = basepath+ s + eachsite + s + eachsensor+s+os.path.split(eachfile)[1] + ".SZA.tif"
                    dataset, gcs, pcs, extend, shape = get_file_info(bandfile)
                    szabanddata = dataset.GetRasterBand(1).ReadAsArray()
                    # szavalue = szabanddata[row, col]
                    tempsza_array = szabanddata[row - 1:row + 2, col - 1:col + 2]
                    szavalue = np.mean(tempsza_array[tempsza_array > 0])*0.01



                    year = filesplit[3][0:4]
                    doy = filesplit[3][4:7]
                    # sensor.append("L30V2")
                    redbandlist.append(faparvalue)
                    nirbandlist.append(faparstdvalue)
                    szabandlist.append(szavalue)
                    yearlist.append(year)
                    DOYlist.append(doy)
                    tilelist.append(eachtile)
                    sensorlist.append(eachsensor)
                    siteoutlist.append(eachsite)
                except Exception as e:
                    print(e)
        # yearlist.append(year)
        # DOYlist.append(doy)
        # tilelist.append(eachtile)
        df["year"]=yearlist
        df["doy"]=DOYlist
        df["tile"]=tilelist
        df["sensor"]=sensorlist
        df["HLSFAPAR"]=redbandlist
        # df["FieldFAPAR"]=fieldFAPARlist
        df["HLSFAPARstd"]=nirbandlist
        df["sza"]=szabandlist
        df["site"]=siteoutlist

        df.to_csv("./Data" + s + eachsite + "_HLS_InsFAPAR.csv")
        print("./Data" + s + eachsite + "_HLS_InsFAPAR.csv")
