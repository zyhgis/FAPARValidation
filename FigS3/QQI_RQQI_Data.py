# -*- coding: utf-8 -*-
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# -*- coding: utf-8 -*-
import os
import numpy as np
from osgeo import osr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import gc
import datetime
from osgeo import gdal
import pandas as pd


class GRID:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件

        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

        del dataset
        return im_proj, im_geotrans, im_data

    # 写文件，以写成tif为例
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset


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


def probav300FAPARlistdir(path):
    list_name = []
    for file in os.listdir(path):
        marker = file.split("_")[5][-5:]
        if os.path.splitext(file)[1] == '.tif' and marker == "FAPAR":
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    return list_name

def probav300RMSElistdir(path):
    list_name = []
    for file in os.listdir(path):
        marker = file.split("_")[5][-4:]
        if os.path.splitext(file)[1] == '.tif' and marker == "RMSE":
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    return list_name
def probav1000FAPARlistdir(path):
    list_name = []
    # CA-TP4_project_CA-TP4_c_gls_FAPAR300-QFLAG_201503100000_CUSTOM_PROBAV_V1.0.1
    # CA-TP4_project_CA-TP4_c_gls_FAPAR-RT6-QFLAG_201802100000_CUSTOM_PROBAV_V2.0.1
    for file in os.listdir(path):
        # print file
        marker = file.split("_")[5][-5:]
        if os.path.splitext(file)[1] == '.tif' and marker == "FAPAR":
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    return list_name

def probav1000RMSElistdir(path):
    list_name = []
    # CA-TP4_project_CA-TP4_c_gls_FAPAR300-QFLAG_201503100000_CUSTOM_PROBAV_V1.0.1
    # CA-TP4_project_CA-TP4_c_gls_FAPAR-RT6-QFLAG_201802100000_CUSTOM_PROBAV_V2.0.1
    for file in os.listdir(path):
        # print file
        marker = file.split("_")[5][-4:]
        if os.path.splitext(file)[1] == '.tif' and marker == "RMSE":
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    return list_name

def MODMYDlistdir(path):
    list_name = []
    # CA-TP4_project_CA-TP4_c_gls_FAPAR300-QFLAG_201503100000_CUSTOM_PROBAV_V1.0.1
    # CA-TP4_project_CA-TP4_MOD15A2HA2000049h12v04FparLai_QC.tif
    for file in os.listdir(path):
        # print file
        marker = file.split("Fpar")[0]
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, marker)
            list_name.append(file_path)
    list_name=sorted(list(set(list_name)))
    return list_name

def VNPlistdir(path):
    list_name = []
    # CA-TP4_project_CA-TP4_c_gls_FAPAR300-QFLAG_201503100000_CUSTOM_PROBAV_V1.0.1
    # CA-TP4_project_VNP15A2H.A2012017.h12v04.FparLai_QC.tif
    for file in os.listdir(path):
        # print file
        marker = file.split("Fpar")[0]
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, marker)
            list_name.append(file_path)
    list_name = sorted(list(set(list_name)))
    return list_name

if __name__ == "__main__":
    s = os.sep
    run = GRID()

    MODSZAPATH = "D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MOD\Site3"
    # MOD_SZA_CA-TP4.csv
    MYDSZAPATH = "D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MYD\Site3"
    VNPSZAPATH = "D:\FAPAR_Validation_NA\FAPARproduct\VNP\Data\FAPAR\Site3"
    # VNP_SZA_CA-TP4.csv
    PROBAV300SZAPATH = "D:\FAPAR_Validation_NA\FAPARproduct\PROBAV\Data\PROBAV300\Site3"
    PROBAV1000SZAPATH = "D:\FAPAR_Validation_NA\FAPARproduct\PROBAV\Data\PROBAV1000\Site3"

    # PROBAV_SZA_CA-TP4.csv
    sitelist = ["CA-TP4", "CA-TPD", "US-Bar", "US-HF"]
    for eachsite in sitelist:
        yearlist=[]
        doylist=[]
        FAPARlist=[]
        FAPARstdlist=[]
        MODSZAfile = MODSZAPATH + s + eachsite
        MODfilelist = MODMYDlistdir(MODSZAfile)
        for eachfile in MODfilelist:
            year = os.path.split(eachfile)[1].split("_")[3][9:13]
            doy = os.path.split(eachfile)[1].split("_")[3][13:16]
            FAPARdata=run.read_img(eachfile+"Fpar_500m.tif")[2]
            FAPARstd=run.read_img(eachfile+"FparStdDev_500m.tif")[2]
            fliter= FAPARstd<101
            FAPARdata_1d=FAPARdata[fliter]
            FAPARstd_1d=FAPARstd[fliter]
            FAPARmean=np.mean(FAPARdata_1d)*0.01
            FAPARstdmean=np.mean(FAPARstd_1d)*0.01
            FAPARlist.append(FAPARmean)
            FAPARstdlist.append(FAPARstdmean)
            yearlist.append(year)
            doylist.append(doy)
        df=pd.DataFrame()
        df["year"]=yearlist
        df['doy']=doylist
        df['FAPAR']=FAPARlist
        df['FAPARstd']=FAPARstdlist
        df.to_csv("./Data"+s+eachsite+"_MODFAPAR.csv")

        yearlist = []
        doylist = []
        FAPARlist = []
        FAPARstdlist = []
        MYDSZAfile = MYDSZAPATH + s + eachsite
        MYDfilelist = MODMYDlistdir(MYDSZAfile)
        for eachfile in MYDfilelist:
            year = os.path.split(eachfile)[1].split("_")[3][9:13]
            doy = os.path.split(eachfile)[1].split("_")[3][13:16]
            FAPARdata = run.read_img(eachfile + "Fpar_500m.tif")[2]
            FAPARstd = run.read_img(eachfile + "FparStdDev_500m.tif")[2]
            fliter = FAPARstd < 101
            FAPARdata_1d = FAPARdata[fliter]
            FAPARstd_1d = FAPARstd[fliter]
            FAPARmean = np.mean(FAPARdata_1d)*0.01
            FAPARstdmean = np.mean(FAPARstd_1d)*0.01
            FAPARlist.append(FAPARmean)
            FAPARstdlist.append(FAPARstdmean)
            yearlist.append(year)
            doylist.append(doy)
        df = pd.DataFrame()
        df["year"] = yearlist
        df['doy'] = doylist
        df['FAPAR'] = FAPARlist
        df['FAPARstd'] = FAPARstdlist
        df.to_csv("./Data" + s + eachsite + "_MYDFAPAR.csv")

        yearlist = []
        doylist = []
        FAPARlist = []
        FAPARstdlist = []
        VNPSZAfile = VNPSZAPATH + s + eachsite
        VNPfilelist = VNPlistdir(VNPSZAfile)
        for eachfile in VNPfilelist:
            year = os.path.split(eachfile)[1].split("_")[2][10:14]
            doy = os.path.split(eachfile)[1].split("_")[2][14:17]
            FAPARdata = run.read_img(eachfile + "Fpar.tif")[2]
            FAPARstd = run.read_img(eachfile + "FparStdDev.tif")[2]
            fliter = FAPARstd < 101
            FAPARdata_1d = FAPARdata[fliter]
            FAPARstd_1d = FAPARstd[fliter]
            FAPARmean = np.mean(FAPARdata_1d)*0.01
            FAPARstdmean = np.mean(FAPARstd_1d)*0.01
            FAPARlist.append(FAPARmean)
            FAPARstdlist.append(FAPARstdmean)
            yearlist.append(year)
            doylist.append(doy)
        df = pd.DataFrame()
        df["year"] = yearlist
        df['doy'] = doylist
        df['FAPAR'] = FAPARlist
        df['FAPARstd'] = FAPARstdlist
        df.to_csv("./Data" + s + eachsite + "_VNPFAPAR.csv")

        yearlist = []
        doylist = []
        FAPARlist = []
        FAPARstdlist = []
        PROBAVfile = PROBAV300SZAPATH + s + eachsite
        probav300RMSElist = probav300RMSElistdir(PROBAVfile)
        probav300FAPARlist = probav300FAPARlistdir(PROBAVfile)
        # CA-TP4_project_CA-TP4_c_gls_FAPAR300-FAPAR_201401100000_CUSTOM_PROBAV_V1.0.1.tif
        for eachfile in probav300FAPARlist:
            temp = os.path.split(eachfile)[1].split("_")[6][:8]
            tempdate = datetime.datetime.strptime(temp, '%Y%m%d')
            year = tempdate.year
            doy = tempdate.timetuple().tm_yday

            FAPARdata = run.read_img(eachfile)[2]
            eachfilename=os.path.split(eachfile)[1]
            eachfilepath=os.path.split(eachfile)[0]
            filesplitlist=eachfilename.split("_")
            RMSEfilelist=filesplitlist[0:5]
            RMSEfilelist.append("FAPAR300-RMSE")
            RMSEfilelist.extend(filesplitlist[6:])
            RMSEfile="_".join(RMSEfilelist)
            if os.path.exists(eachfilepath+s+RMSEfile)==False:
                continue
            FAPARstd = run.read_img(eachfilepath+s+RMSEfile)[2]
            fliter = FAPARstd < 236
            FAPARdata_1d = FAPARdata[fliter]
            FAPARstd_1d = FAPARstd[fliter]
            FAPARmean = np.mean(FAPARdata_1d)/250
            FAPARstdmean = np.mean(FAPARstd_1d)/250
            FAPARlist.append(FAPARmean)
            FAPARstdlist.append(FAPARstdmean)
            yearlist.append(year)
            doylist.append(doy)
        df = pd.DataFrame()
        df["year"] = yearlist
        df['doy'] = doylist
        df['FAPAR'] = FAPARlist
        df['FAPARstd'] = FAPARstdlist
        df.to_csv("./Data" + s + eachsite + "_PROBAV300FAPAR.csv")

        yearlist = []
        doylist = []
        FAPARlist = []
        FAPARstdlist = []
        PROBAVfile = PROBAV1000SZAPATH + s + eachsite
        probav1000FAPARlist = probav1000FAPARlistdir(PROBAVfile)
        # CA-TP4_project_CA-TP4_c_gls_FAPAR300-FAPAR_201401100000_CUSTOM_PROBAV_V1.0.1.tif
        for eachfile in probav1000FAPARlist:
            temp = os.path.split(eachfile)[1].split("_")[6][:8]
            tempdate = datetime.datetime.strptime(temp, '%Y%m%d')
            year = tempdate.year
            doy = tempdate.timetuple().tm_yday

            FAPARdata = run.read_img(eachfile)[2]
            eachfilename = os.path.split(eachfile)[1]
            eachfilepath = os.path.split(eachfile)[0]
            filesplitlist = eachfilename.split("_")
            RMSEfilelist = filesplitlist[0:5]
            RMSEfilelist.append("FAPAR-RT6-RMSE")
            RMSEfilelist.extend(filesplitlist[6:])
            RMSEfile = "_".join(RMSEfilelist)
            if os.path.exists(eachfilepath + s + RMSEfile) == False:
                continue

            FAPARstd = run.read_img(eachfilepath + s + RMSEfile)[2]
            fliter = FAPARstd < 236
            FAPARdata_1d = FAPARdata[fliter]
            FAPARstd_1d = FAPARstd[fliter]
            FAPARmean = np.mean(FAPARdata_1d)/250
            FAPARstdmean = np.mean(FAPARstd_1d)/250
            FAPARlist.append(FAPARmean)
            FAPARstdlist.append(FAPARstdmean)
            yearlist.append(year)
            doylist.append(doy)
        df = pd.DataFrame()
        df["year"] = yearlist
        df['doy'] = doylist
        df['FAPAR'] = FAPARlist
        df['FAPARstd'] = FAPARstdlist
        df.to_csv("./Data" + s + eachsite + "_PROBAV1000FAPAR.csv")
