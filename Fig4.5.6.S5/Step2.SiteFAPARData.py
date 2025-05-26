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

if __name__ == '__main__':
    s=os.sep
    tilesdict = {"CA-TP4":["T17TNH"], "CA-TPD":["T17TNH"],"US-Bar": ["T18TYP", "T19TCJ"], "US-HF": ["T18TYN"],
                 "US-Prr": ["T06WVT"], "US-Uaf": ["T06WVS", "T06WVT"]}
    latdict = {"CA-TP4":42.7102, "CA-TPD":42.6353,"US-Bar": 44.0646, "US-HF": 42.5353,
                 "US-Prr": 65.1237, "US-Uaf": 64.8663}
    londict = {"CA-TP4":-80.3574, "CA-TPD":-80.5577,"US-Bar": -71.2881, "US-HF": -72.1899,
                 "US-Prr": -147.4876, "US-Uaf": -147.8555}

    sitelist = ["CA-TP4", "CA-TPD", "US-Bar", "US-HF"]
    fieldPATH="D:\FAPAR_Validation_NA\RetrieveFAPAR\ClearValidationData"

    FAPARpath=r"D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MOD\Site3"

    df = pd.DataFrame()
    yearlist = []
    DOYlist = []
    fieldFAPARlist = []
    RSFAPARlist=[]
    siteoutlist=[]
    for eachsite in sitelist:
        fieldcsv = fieldPATH + s + eachsite + "-Clear-Clear.csv"
        fielddf=pd.read_csv(fieldcsv)

        lat=latdict[eachsite]
        lon=londict[eachsite]
        coordinates=[lon,lat]


        filepath = FAPARpath + s + eachsite
        Tiflist = listdir(filepath)
        # D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MOD\Site3\CA-TP4\CA-TP4_project_CA-TP4_MOD15A2HA2000049h12v04Fpar_500m.tif
        for eachfile in Tiflist:
            eachsensor = os.path.split(eachfile)[1][-13:-4]
            if eachsensor == "Fpar_500m":
                tempyear=int(os.path.split(eachfile)[1][-26:-22])
                tempdoy=int(os.path.split(eachfile)[1][-22:-19])
                dataset, gcs, pcs, extend, shape = get_file_info(eachfile)
                x, y, _ = lonlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
                row, col = xy_to_rowcol(extend, x, y)

                fapardata = dataset.GetRasterBand(1).ReadAsArray()
                tempfapar_array = fapardata[row-1:row+2, col-1:col+2]
                tempfapar_array=tempfapar_array[tempfapar_array > 0]
                tempfapar_array=tempfapar_array[tempfapar_array < 100]
                faparvalue=np.mean(tempfapar_array[tempfapar_array>0])
                faparvaluelen=len(tempfapar_array[tempfapar_array>0])
                if faparvaluelen==0:
                    faparvalue=0
                if faparvalue==0:
                    continue
                bandfile = eachfile
                dataset, gcs, pcs, extend, shape = get_file_info(bandfile)
                faparstddata = dataset.GetRasterBand(1).ReadAsArray()
                tempfaparstd_array = faparstddata[row - 1:row + 2, col - 1:col + 2]

                faparstdvalue = np.mean(tempfaparstd_array[tempfaparstd_array > 0])
                # faparsvaluelen = len(tempfaparstd_array[tempfaparstd_array > 0])
                if faparvaluelen == 0:
                    faparstdvalue = 0

                tempfielddf = fielddf[(fielddf["HOURMIN"] >= 9) &(fielddf["HOURMIN"] < 11) & (fielddf["year"] == tempyear) & (
                            fielddf["DOY"] >= tempdoy - 8) & (fielddf["DOY"] <= tempdoy)]

                if len(tempfielddf) == 0:
                    print(eachsite, tempyear, tempdoy)
                    continue

                tempfapar = np.mean(tempfielddf["FAPAR"].values.tolist())

                RSFAPARlist.append(faparvalue*0.01)
                fieldFAPARlist.append(tempfapar)
                yearlist.append(tempyear)
                DOYlist.append(tempdoy)
                siteoutlist.append(eachsite)
    df["year"]=yearlist
    df["doy"]=DOYlist
    df["RSFAPAR"]=RSFAPARlist
    df["FieldFAPAR"]=fieldFAPARlist
    df["site"]=siteoutlist
    df.to_csv("MOD_Field_FAPAR.csv")

    FAPARpath = r"D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MYD\Site3"

    df = pd.DataFrame()
    yearlist = []
    DOYlist = []
    fieldFAPARlist = []
    RSFAPARlist = []
    siteoutlist = []
    for eachsite in sitelist:
        fieldcsv = fieldPATH + s + eachsite + "-Clear-Clear.csv"
        fielddf = pd.read_csv(fieldcsv)

        lat = latdict[eachsite]
        lon = londict[eachsite]
        coordinates = [lon, lat]

        filepath = FAPARpath + s + eachsite
        Tiflist = listdir(filepath)
        # D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MOD\Site3\CA-TP4\CA-TP4_project_CA-TP4_MOD15A2HA2000049h12v04Fpar_500m.tif
        for eachfile in Tiflist:
            eachsensor = os.path.split(eachfile)[1][-13:-4]
            if eachsensor == "Fpar_500m":
                tempyear = int(os.path.split(eachfile)[1][-26:-22])
                tempdoy = int(os.path.split(eachfile)[1][-22:-19])
                dataset, gcs, pcs, extend, shape = get_file_info(eachfile)
                x, y, _ = lonlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
                row, col = xy_to_rowcol(extend, x, y)

                fapardata = dataset.GetRasterBand(1).ReadAsArray()
                tempfapar_array = fapardata[row - 1:row + 2, col - 1:col + 2]
                tempfapar_array=tempfapar_array[tempfapar_array > 0]
                tempfapar_array=tempfapar_array[tempfapar_array < 100]
                faparvalue = np.mean(tempfapar_array[tempfapar_array > 0])
                faparvaluelen = len(tempfapar_array[tempfapar_array > 0])
                if faparvaluelen == 0:
                    faparvalue = 0
                if faparvalue == 0:
                    continue
                bandfile = eachfile
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

                tempfielddf = fielddf[
                    (fielddf["HOURMIN"] >= 13) & (fielddf["HOURMIN"] < 14) & (fielddf["year"] == tempyear) & (
                            fielddf["DOY"] >= tempdoy - 8) & (fielddf["DOY"] <= tempdoy)]

                if len(tempfielddf) == 0:
                    print(eachsite, tempyear, tempdoy)
                    continue

                tempfapar = np.mean(tempfielddf["FAPAR"].values.tolist())

                RSFAPARlist.append(faparvalue*0.01)
                fieldFAPARlist.append(tempfapar)
                yearlist.append(tempyear)
                DOYlist.append(tempdoy)
                siteoutlist.append(eachsite)
    df["year"] = yearlist
    df["doy"] = DOYlist
    df["RSFAPAR"] = RSFAPARlist
    df["FieldFAPAR"] = fieldFAPARlist
    df["site"] = siteoutlist
    df.to_csv("MYD_Field_FAPAR.csv")

    FAPARpath = r"D:\FAPAR_Validation_NA\FAPARproduct\VNP\Data\FAPAR\Site3"

    df = pd.DataFrame()
    yearlist = []
    DOYlist = []
    fieldFAPARlist = []
    RSFAPARlist = []
    siteoutlist = []
    for eachsite in sitelist:
        fieldcsv = fieldPATH + s + eachsite + "-Clear-Clear.csv"
        fielddf = pd.read_csv(fieldcsv)

        lat = latdict[eachsite]
        lon = londict[eachsite]
        coordinates = [lon, lat]

        filepath = FAPARpath + s + eachsite
        Tiflist = listdir(filepath)
        # D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MOD\Site3\CA-TP4\CA-TP4_project_CA-TP4_MOD15A2HA2000049h12v04Fpar_500m.tif
        for eachfile in Tiflist:
            eachsensor = os.path.split(eachfile)[1][-8:-4]
            # CA-TP4_project_VNP15A2H.A2020329.h12v04.Fpar.tif
            if eachsensor == "Fpar":
                tempyear = int(os.path.split(eachfile)[1][-23:-19])
                tempdoy = int(os.path.split(eachfile)[1][-19:-16])
                dataset, gcs, pcs, extend, shape = get_file_info(eachfile)
                x, y, _ = lonlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
                row, col = xy_to_rowcol(extend, x, y)

                fapardata = dataset.GetRasterBand(1).ReadAsArray()
                tempfapar_array = fapardata[row - 1:row + 2, col - 1:col + 2]
                tempfapar_array=tempfapar_array[tempfapar_array > 0]
                tempfapar_array=tempfapar_array[tempfapar_array < 100]

                faparvalue = np.mean(tempfapar_array[tempfapar_array > 0])
                faparvaluelen = len(tempfapar_array[tempfapar_array > 0])
                if faparvaluelen == 0:
                    faparvalue = 0
                if faparvalue == 0:
                    continue
                bandfile = eachfile
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

                tempfielddf = fielddf[
                    (fielddf["HOURMIN"] >= 13) & (fielddf["HOURMIN"] < 14) & (fielddf["year"] == tempyear) & (
                            fielddf["DOY"] >= tempdoy - 8) & (fielddf["DOY"] <= tempdoy)]

                if len(tempfielddf) == 0:
                    print(eachsite, tempyear, tempdoy)
                    continue

                tempfapar = np.mean(tempfielddf["FAPAR"].values.tolist())

                RSFAPARlist.append(faparvalue * 0.01)
                fieldFAPARlist.append(tempfapar)
                yearlist.append(tempyear)
                DOYlist.append(tempdoy)
                siteoutlist.append(eachsite)
    df["year"] = yearlist
    df["doy"] = DOYlist
    df["RSFAPAR"] = RSFAPARlist
    df["FieldFAPAR"] = fieldFAPARlist
    df["site"] = siteoutlist
    df.to_csv("VNP_Field_FAPAR.csv")


    FAPARpath = r"D:\FAPAR_Validation_NA\FAPARproduct\PROBAV\Data\PROBAV300\Site3"

    df = pd.DataFrame()
    yearlist = []
    DOYlist = []
    fieldFAPARlist = []
    RSFAPARlist = []
    siteoutlist = []
    for eachsite in sitelist:
        fieldcsv = fieldPATH + s + eachsite + "-Clear-Clear.csv"
        fielddf = pd.read_csv(fieldcsv)

        lat = latdict[eachsite]
        lon = londict[eachsite]
        coordinates = [lon, lat]

        filepath = FAPARpath + s + eachsite
        Tiflist = listdir(filepath)
        # D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MOD\Site3\CA-TP4\CA-TP4_project_CA-TP4_MOD15A2HA2000049h12v04Fpar_500m.tif
        # CA-TP4_project_CA-TP4_c_gls_FAPAR300-FAPAR_201401100000_CUSTOM_PROBAV_V1.0.1.tif
        for eachfile in Tiflist:
            eachsensor = os.path.split(eachfile)[1][-13:-4]
            filename=os.path.split(eachfile)[1]
            namesplitlist=filename.split("_")

            # CA-TP4_project_CA-TP4_c_gls_FAPAR300-FAPAR_201401100000_CUSTOM_PROBAV_V1.0.1.tif
            if namesplitlist[5] == "FAPAR300-FAPAR":
                tempyear = int(namesplitlist[6][0:4])
                tempmonth = int(namesplitlist[6][4:6])
                tempday = int(namesplitlist[6][6:8])
                tempdoy=(tempmonth-1)*30+tempday

                dataset, gcs, pcs, extend, shape = get_file_info(eachfile)
                x, y, _ = lonlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
                row, col = xy_to_rowcol(extend, x, y)

                fapardata = dataset.GetRasterBand(1).ReadAsArray()
                tempfapar_array = fapardata[row - 1:row + 2, col - 1:col + 2]
                tempfapar_array=tempfapar_array[tempfapar_array > 0]
                tempfapar_array=tempfapar_array[tempfapar_array < 250]
                faparvalue = np.mean(tempfapar_array[tempfapar_array > 0])
                faparvaluelen = len(tempfapar_array[tempfapar_array > 0])
                if faparvaluelen == 0:
                    faparvalue = 0
                if faparvalue == 0:
                    continue
                bandfile = eachfile
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

                tempfielddf = fielddf[
                    (fielddf["HOURMIN"] >= 9) & (fielddf["HOURMIN"] < 11) & (fielddf["year"] == tempyear) & (
                            fielddf["DOY"] >= tempdoy - 10) & (fielddf["DOY"] <= tempdoy)]

                if len(tempfielddf) == 0:
                    print(eachsite, tempyear, tempdoy)
                    continue

                tempfapar = np.mean(tempfielddf["FAPAR"].values.tolist())

                RSFAPARlist.append(faparvalue/250)
                fieldFAPARlist.append(tempfapar)
                yearlist.append(tempyear)
                DOYlist.append(tempdoy)
                siteoutlist.append(eachsite)
    df["year"] = yearlist
    df["doy"] = DOYlist
    df["RSFAPAR"] = RSFAPARlist
    df["FieldFAPAR"] = fieldFAPARlist
    df["site"] = siteoutlist
    df.to_csv("PROBAV300_Field_FAPAR.csv")

    FAPARpath = r"D:\FAPAR_Validation_NA\FAPARproduct\PROBAV\Data\PROBAV1000\Site3"

    df = pd.DataFrame()
    yearlist = []
    DOYlist = []
    fieldFAPARlist = []
    RSFAPARlist = []
    siteoutlist = []
    for eachsite in sitelist:
        fieldcsv = fieldPATH + s + eachsite + "-Clear-Clear.csv"
        fielddf = pd.read_csv(fieldcsv)

        lat = latdict[eachsite]
        lon = londict[eachsite]
        coordinates = [lon, lat]

        filepath = FAPARpath + s + eachsite
        Tiflist = listdir(filepath)
        # D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MOD\Site3\CA-TP4\CA-TP4_project_CA-TP4_MOD15A2HA2000049h12v04Fpar_500m.tif
        # CA-TP4_project_CA-TP4_c_gls_FAPAR300-FAPAR_201401100000_CUSTOM_PROBAV_V1.0.1.tif
        for eachfile in Tiflist:
            eachsensor = os.path.split(eachfile)[1][-13:-4]
            filename = os.path.split(eachfile)[1]
            namesplitlist = filename.split("_")

            # CA-TP4_project_CA-TP4_c_gls_FAPAR-RT6-FAPAR_201703100000_CUSTOM_PROBAV_V2.0.2.tif
            if namesplitlist[5] == "FAPAR-RT6-FAPAR":
                tempyear = int(namesplitlist[6][0:4])
                tempmonth = int(namesplitlist[6][4:6])
                tempday = int(namesplitlist[6][6:8])
                tempdoy = (tempmonth - 1) * 30 + tempday

                dataset, gcs, pcs, extend, shape = get_file_info(eachfile)
                x, y, _ = lonlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
                row, col = xy_to_rowcol(extend, x, y)

                fapardata = dataset.GetRasterBand(1).ReadAsArray()
                tempfapar_array = fapardata[row - 1:row + 2, col - 1:col + 2]
                tempfapar_array=tempfapar_array[tempfapar_array > 0]
                tempfapar_array=tempfapar_array[tempfapar_array < 250]
                faparvalue = np.mean(tempfapar_array[tempfapar_array > 0])
                faparvaluelen = len(tempfapar_array[tempfapar_array > 0])
                if faparvaluelen == 0:
                    faparvalue = 0
                if faparvalue == 0:
                    continue
                bandfile = eachfile
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

                tempfielddf = fielddf[
                    (fielddf["HOURMIN"] >= 9) & (fielddf["HOURMIN"] < 11) & (fielddf["year"] == tempyear) & (
                            fielddf["DOY"] >= tempdoy - 10) & (fielddf["DOY"] <= tempdoy)]

                if len(tempfielddf) == 0:
                    print(eachsite, tempyear, tempdoy)
                    continue

                tempfapar = np.mean(tempfielddf["FAPAR"].values.tolist())

                RSFAPARlist.append(faparvalue/250)
                fieldFAPARlist.append(tempfapar)
                yearlist.append(tempyear)
                DOYlist.append(tempdoy)
                siteoutlist.append(eachsite)
    df["year"] = yearlist
    df["doy"] = DOYlist
    df["RSFAPAR"] = RSFAPARlist
    df["FieldFAPAR"] = fieldFAPARlist
    df["site"] = siteoutlist
    df.to_csv("PROBAV1000_Field_FAPAR.csv")
