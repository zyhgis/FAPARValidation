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


def doy2date(year, doy):
    month_leapyear = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_notleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        for i in range(0, 12):
            if doy > month_leapyear[i]:
                doy -= month_leapyear[i]
                continue
            if doy <= month_leapyear[i]:
                month = i + 1
                day = doy
                break
    else:
        for i in range(0, 12):
            if doy > month_notleap[i]:
                doy -= month_notleap[i]
                continue
            if doy <= month_notleap[i]:
                month = i + 1
                day = doy
                break
    return month, day


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
    list_name = []
    for file in os.listdir(path):
        # print file
        if os.path.splitext(file)[1] == '.tif':
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    return list_name


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
    list_name = sorted(list(set(list_name)))
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


def HLSFAPAR(path):
    list_name = []
    # MODIS.CA-TP4.HLS.L30.T17TNH.2013196T160546.v2.0.FAPAR.tif
    for file in os.listdir(path):
        # print file
        marker = file.split(".")[-2]
        if os.path.splitext(file)[1] == '.tif' and marker == "FAPAR":
            file_path = os.path.join(path, file)
            list_name.append(file_path)
    list_name = sorted(list(set(list_name)))
    return list_name


base = [str(x) for x in range(10)] + [chr(x) for x in range(ord('A'), ord('A') + 6)]


# dec2bin
# 十进制 to 二进制: bin()
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


def Stastic_func(baseim_geotrans, basestartim_data,im_geotrans, startim_data):

    basesize = basestartim_data.shape
    # print(baseim_geotrans)
    basestartlon = baseim_geotrans[0]
    basesp = baseim_geotrans[1]
    basestartlat = baseim_geotrans[3]
    result=np.zeros(basesize)


    ecostartlon = im_geotrans[0]
    ecosp = im_geotrans[1]
    ecostartlat = im_geotrans[3]
    tuple01 = startim_data.shape
    row = tuple01[0]  # 取出数据的行数，赋值给row
    column = tuple01[1]
    for icolumn in range(0, basesize[1]):
        tempstartlon = basestartlon + icolumn * basesp
        tempendlon = tempstartlon + basesp
        # temp_eco_startlon=ecostartlon+icolumn*basesp
        startx = int(round((tempstartlon - ecostartlon) / ecosp, 0))
        if startx < 0:
            startx = 0
        if icolumn == basesize[0] - 1:
            endx = column
        else:
            endx = int(round((tempendlon - ecostartlon) / ecosp, 0))
        for irow in range(0, basesize[0]):
            tempstartlat = basestartlat - irow * basesp
            tempendlat = tempstartlat - basesp
            # temp_eco_startlon=ecostartlon+icolumn*basesp
            starty = int(round((ecostartlat - tempstartlat) / ecosp, 0))
            if starty < 0:
                starty = 0
            if irow == basesize[1] - 1:
                endy = row
            else:
                endy = int(round((ecostartlat - tempendlat) / ecosp, 0))

            temp_eco_data = startim_data[starty: endy, startx:endx]
            tuple01 = temp_eco_data.shape
            temprow = tuple01[0]  # 取出数据的行数，赋值给row
            tempcolumn = tuple01[1]
            tempcount = temprow * tempcolumn
            temp_eco_data = temp_eco_data.ravel()
            temp_eco_data=temp_eco_data[temp_eco_data>0]
            validcount = len(temp_eco_data)
            if tempcount>0:
                if validcount/tempcount>0.95:
                    value=np.mean(temp_eco_data)*validcount/tempcount
                else:
                    value = -1
            else:
                value=-1
            result[irow][icolumn] = value
    return result


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    # cols = data.shape[0] # 0表示行数
    # cols1 = data.shape[1] # 1表示列数
    r, h = array.shape

    buchongr = nrows - r % nrows
    if buchongr % 2 == 0:
        addr1 = array[:buchongr // 2, :]
        addr2 = array[buchongr // 2 * -1:, :]
        array = np.row_stack((addr1, array))
        array = np.row_stack((array, addr2))
    else:
        addr1 = array[:buchongr // 2 + 1, :]
        addr2 = array[buchongr // 2 * -1:, :]
        array = np.row_stack((addr1, array))
        array = np.row_stack((array, addr2))

    buchongh = ncols - h % ncols
    if buchongh % 2 == 0:
        addh1 = array[:,:buchongh // 2]
        addh2 = array[:, buchongh // 2 * -1:]
        array = np.column_stack((addh1, array))
        array = np.column_stack((array, addh2))
    else:
        addh1 = array[:, :buchongh // 2 + 1]
        addh2 = array[:, buchongh // 2 * -1:]
        array = np.column_stack((addh1, array))
        array = np.column_stack((array, addh2))

    r, h = array.shape
    subarray = []
    for j in range(nrows):
        for i in range(ncols):
            subarray.append(array[j*int(r / nrows):(j + 1)*int(r / nrows), i*int(h / ncols):(i+1)*int(h / ncols)])

    subarray = np.array(subarray)

    return subarray


if __name__ == "__main__":
    s = os.sep
    run = GRID()

    HLSMODISPATH = r"D:\FAPAR_Validation_NA\2024\FAPAR\Result2\MODIS"
    HLSPROBAV300PATH = r"D:\FAPAR_Validation_NA\2024\FAPAR\Result2\PROBAV300"
    HLSPROBAV1000PATH = r"D:\FAPAR_Validation_NA\2024\FAPAR\Result2\PROBAV1000"

    MODPATH = "D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MOD\Site3"
    MYDPATH = "D:\FAPAR_Validation_NA\FAPARproduct\MODMYD\Data\MYD\Site3"
    VNPPATH = "D:\FAPAR_Validation_NA\FAPARproduct\VNP\Data\FAPAR\Site3"
    PROBAV300PATH = "D:\FAPAR_Validation_NA\FAPARproduct\PROBAV\Data\PROBAV300\Site3"
    PROBAV1000PATH = "D:\FAPAR_Validation_NA\FAPARproduct\PROBAV\Data\PROBAV1000\Site3"

    sitelist = ["US-HF", "CA-TP4", "CA-TPD", "US-Bar"]
    # sitelist = ["US-Uaf"]
    shppath = "D:\BaiduSyncdisk\Paper\FAPAR_Validation_NA\Fig\Shp"

    for eachsite in sitelist:
        print(eachsite)

        # if eachsite=="US-HF":
        #     print("")

        MODtemplate_dataset = shppath + s + eachsite + "-MODIS.tif"
        MODdata = run.read_img(MODtemplate_dataset)[2]
        shapelen = MODdata.shape
        Yearlist = []
        Doylist = []
        index = 0
        cloumnnames = ["year", "doy"]
        for i in range(shapelen[0]):
            for j in range(shapelen[1]):
                index = index + 1
                cloumnnames.append("HLS" + str(index))

                cloumnnames.append("MODFAPAR" + str(index))
                cloumnnames.append("MODSTD" + str(index))
                cloumnnames.append("MODQA" + str(index))

                cloumnnames.append("MYDFAPAR" + str(index))
                cloumnnames.append("MYDSTD" + str(index))
                cloumnnames.append("MYDQA" + str(index))

                cloumnnames.append("VNPFAPAR" + str(index))
                cloumnnames.append("VNPSTD" + str(index))
                cloumnnames.append("VNPQA" + str(index))
        MODISdf = pd.DataFrame(columns=(cloumnnames))
        filelist = HLSFAPAR(HLSMODISPATH + s + eachsite)
        for eachfile in filelist:
            # print(eachfile)
            # MODIS.CA-TP4.HLS.L30.T17TNH.2013196T160546.v2.0.FAPAR.tif
            YEARDOY = os.path.split(eachfile)[1].split(".")[4][:7]
            year = YEARDOY[0:4]
            doy = YEARDOY[4:7]
            MODISDOY = int((int(doy) - 1) / 8) * 8 + 1

            MODFAPARfile = MODPATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_MOD15A2HA" + str(
                year) + str(
                MODISDOY).zfill(3) + "h12v04Fpar_500m.tif"
            MYDFAPARfile = MYDPATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_MYD15A2HA" + str(
                year) + str(
                MODISDOY).zfill(3) + "h12v04Fpar_500m.tif"
            VNPFAPARfile = VNPPATH + s + eachsite + s + eachsite + "_project_VNP15A2H.A" + str(year) + str(
                MODISDOY).zfill(
                3) + ".h12v04.Fpar.tif"

            MODFAPARSTDfile = MODPATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_MOD15A2HA" + str(
                year) + str(
                MODISDOY).zfill(3) + "h12v04FparStdDev_500m.tif"
            MYDFAPARSTDfile = MYDPATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_MYD15A2HA" + str(
                year) + str(
                MODISDOY).zfill(3) + "h12v04FparStdDev_500m.tif"
            VNPFAPARSTDfile = VNPPATH + s + eachsite + s + eachsite + "_project_VNP15A2H.A" + str(year) + str(
                MODISDOY).zfill(
                3) + ".h12v04.FparStdDev.tif"

            MODFAPARQAfile = MODPATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_MOD15A2HA" + str(
                year) + str(
                MODISDOY).zfill(3) + "h12v04FparLai_QC.tif"
            MYDFAPARQAfile = MYDPATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_MYD15A2HA" + str(
                year) + str(
                MODISDOY).zfill(3) + "h12v04FparLai_QC.tif"
            VNPFAPARQAfile = VNPPATH + s + eachsite + s + eachsite + "_project_VNP15A2H.A" + str(year) + str(
                MODISDOY).zfill(
                3) + ".h12v04.FparLai_QC.tif"

            HLSMODISdataall = run.read_img(eachfile)
            im_geotrans,HLSMODISdata = HLSMODISdataall[1],HLSMODISdataall[2]

            if os.path.exists(MODFAPARfile)==False:
                continue
            MODFAPARdataall = run.read_img(MODFAPARfile)
            baseim_geotrans,MODFAPARdata = MODFAPARdataall[1],MODFAPARdataall[2]*0.01

            MYDFAPARdata = run.read_img(MYDFAPARfile)[2]*0.01
            VNPFAPARdata = run.read_img(VNPFAPARfile)[2]*0.01

            MODFAPARSTDdata = run.read_img(MODFAPARSTDfile)[2]*0.01
            MYDFAPARSTDdata = run.read_img(MYDFAPARSTDfile)[2]*0.01
            VNPFAPARSTDdata = run.read_img(VNPFAPARSTDfile)[2]*0.01

            MODFAPARQAdata = run.read_img(MODFAPARQAfile)[2]
            MYDFAPARQAdata = run.read_img(MYDFAPARQAfile)[2]
            VNPFAPARQAdata = run.read_img(VNPFAPARQAfile)[2]
            print(MODFAPARfile)
            shapelen = MODFAPARdata.shape
            # MODHLS = split(HLSMODISdata, shapelen[0], shapelen[1])
            MODHLS=Stastic_func(baseim_geotrans,MODFAPARdata,im_geotrans,HLSMODISdata)
            rowlist = []
            rowlist.append(year)
            rowlist.append(MODISDOY)
            for i in range(shapelen[0]):
                for j in range(shapelen[1]):

                    tempmean = MODHLS[i][j]
                    if tempmean==-1:
                        rowlist.extend([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
                        continue
                    # shapesum = tempdata.shape
                    # shapesum=shapesum[0]*shapesum[1]
                    # tempdata = tempdata[tempdata > 0]
                    # if len(tempdata) / shapesum < 0.95:
                    #     rowlist.extend([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
                    #     continue

                    # tempmean = np.mean(tempdata)
                    rowlist.append(tempmean)
                    MODvalue = MODFAPARdata[i][j]
                    if MODvalue>1:
                        MODvalue=-1
                    rowlist.append(MODvalue)
                    # if eachsite=="US-HF" and MODvalue>0.9 and tempmean<0.5:
                    #     print(MODvalue)
                    MODstdvalue = MODFAPARSTDdata[i][j]
                    if MODstdvalue>1:
                        MODstdvalue=-1
                    rowlist.append(MODstdvalue)
                    MODQAvalue = MODFAPARQAdata[i][j]
                    QA = dec2bin8(MODQAvalue)
                    QAvalue=0
                    if QA[0:3] == "000":  # Main(RT)methodused,bestresultpossible(nosaturation
                        QAvalue = 1
                    elif QA[0:3] == "001":  # Main(RT)method used with saturation
                        QAvalue = 2
                    elif QA[0:3] == "010":  # bad geometry empirical algorithm used
                        QAvalue = 3
                    elif QA[0:3] == "011":  # due to problems other than geometryempirical algorithm used
                        QAvalue = 4
                    rowlist.append(QAvalue)

                    MYDvalue = MYDFAPARdata[i][j]
                    if MYDvalue>1:
                        MYDvalue=-1
                    rowlist.append(MYDvalue)
                    MYDstdvalue = MYDFAPARSTDdata[i][j]
                    if MYDstdvalue>1:
                        MYDvalue=-1
                    rowlist.append(MYDstdvalue)
                    MYDQAvalue = MYDFAPARQAdata[i][j]
                    QA = dec2bin8(MYDQAvalue)
                    QAvalue=0
                    if QA[0:3] == "000":  # Main(RT)methodused,bestresultpossible(nosaturation
                        QAvalue = 1
                    elif QA[0:3] == "001":  # Main(RT)method used with saturation
                        QAvalue = 2
                    elif QA[0:3] == "010":  # bad geometry empirical algorithm used
                        QAvalue = 3
                    elif QA[0:3] == "011":  # due to problems other than geometryempirical algorithm used
                        QAvalue = 4
                    rowlist.append(QAvalue)

                    VNPvalue = VNPFAPARdata[i][j]
                    if VNPvalue>1:
                        VNPvalue=-1
                    rowlist.append(VNPvalue)
                    VNPstdvalue = VNPFAPARSTDdata[i][j]
                    if VNPstdvalue>1:
                        VNPstdvalue=-1
                    rowlist.append(VNPstdvalue)
                    VNPQAvalue = VNPFAPARQAdata[i][j]
                    QA = dec2bin8(VNPQAvalue)
                    QAvalue=0
                    if QA[-3:] == "000":  # Main(RT)methodused,bestresultpossible(nosaturation
                        QAvalue = 1
                    elif QA[-3:] == "001":  # Main(RT)method used with saturation
                        QAvalue = 2
                    elif QA[-3:] == "010":  # bad geometry empirical algorithm used
                        QAvalue = 3
                    elif QA[-3:] == "011":  # due to problems other than geometryempirical algorithm used
                        QAvalue = 4
                    rowlist.append(QAvalue)
            MODISdf.loc[len(MODISdf)] = rowlist
        MODISdf.to_csv("./Data"+s+eachsite+"_HLS_MODIS_FAPAR_RMSE_QC.csv")

        print("./Data"+s+eachsite+"_HLS_MODIS_FAPAR_RMSE_QC.csv")


        P300template_dataset = shppath + s + eachsite + "-PROBAV300.tif"
        MODdata = run.read_img(P300template_dataset)[2]
        shapelen = MODdata.shape
        Yearlist = []
        Doylist = []
        index = 0
        cloumnnames = ["year", "doy"]
        for i in range(shapelen[0]):
            for j in range(shapelen[1]):
                index = index + 1
                cloumnnames.append("HLS" + str(index))
                cloumnnames.append("PROBAV300FAPAR" + str(index))
                cloumnnames.append("PROBAV300STD" + str(index))
                cloumnnames.append("PROBAV300QA" + str(index))

        MODISdf = pd.DataFrame(columns=(cloumnnames))

        filelist = HLSFAPAR(HLSPROBAV300PATH + s + eachsite)

        for eachfile in filelist:
            # print(eachfile)
            # MODIS.CA-TP4.HLS.L30.T17TNH.2013196T160546.v2.0.FAPAR.tif
            YEARDOY = os.path.split(eachfile)[1].split(".")[4][:7]
            year = YEARDOY[0:4]
            doy = YEARDOY[4:7]
            month, day = doy2date(int(year), int(doy))
            if day <= 10:
                GEODAY = 10
            elif day > 20:
                GEODAY = 30
            else:
                GEODAY = 20
            GEOMODTHDAY = str(year) + str(month).zfill(2) + str(GEODAY).zfill(2)
            PROBAV300FAPARfile = PROBAV300PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR300-FAPAR_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V1.0.1.tif"
            if os.path.exists(PROBAV300FAPARfile)==False:
                continue
            PROBAV300RMSEfile = PROBAV300PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR300-RMSE_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V1.0.1.tif"
            PROBAV300QCfile = PROBAV300PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR300-QFLAG_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V1.0.1.tif"

            if day > 20:
                for GEODAY in [28, 29, 30, 31]:
                    GEOMODTHDAY = str(year) + str(month).zfill(2) + str(GEODAY).zfill(2)
                    PROBAV300FAPARfile = PROBAV300PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR300-FAPAR_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V1.0.1.tif"
                    PROBAV300RMSEfile = PROBAV300PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR300-RMSE_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V1.0.1.tif"
                    PROBAV300QCfile = PROBAV300PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR300-QFLAG_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V1.0.1.tif"
                    if os.path.exists(PROBAV300FAPARfile):
                        break

            HLSP300dataall = run.read_img(eachfile)
            im_geotrans,HLSP300data = HLSP300dataall[1],HLSP300dataall[2]
            P3FAPARdataall = run.read_img(PROBAV300FAPARfile)
            baseim_geotrans,P3FAPARdata = P3FAPARdataall[1],P3FAPARdataall[2]/250

            P3RMSEdata = run.read_img(PROBAV300RMSEfile)[2]/250
            P3QCdata = run.read_img(PROBAV300QCfile)[2]

            shapelen = P3FAPARdata.shape
            # P300HLS = split(HLSP300data, shapelen[0], shapelen[1])
            P300HLS=Stastic_func(baseim_geotrans,P3FAPARdata,im_geotrans,HLSP300data)

            rowlist = []
            rowlist.append(year)
            tempdate = datetime.datetime.strptime(GEOMODTHDAY, '%Y%m%d')
            doy = tempdate.timetuple().tm_yday
            rowlist.append(doy)
            # rowlist.append(GEOMODTHDAY)
            for i in range(shapelen[0]):
                for j in range(shapelen[1]):
                    # hi=i*shapelen[0]+j
                    tempmean = P300HLS[i][j]
                    if tempmean==-1:
                        rowlist.extend([-1, -1, -1, -1])
                        continue
                    # shapesum = tempdata.shape
                    # shapesum=shapesum[0]*shapesum[1]
                    # tempdata = tempdata[tempdata > 0]
                    # if len(tempdata) / shapesum < 0.95:
                    #     rowlist.extend([-1, -1, -1, -1])
                    #     continue

                    # tempmean = np.mean(tempdata)
                    rowlist.append(tempmean)
                    MODvalue = P3FAPARdata[i][j]
                    if MODvalue>1:
                        MODvalue=-1
                    rowlist.append(MODvalue)
                    MODstdvalue = P3RMSEdata[i][j]
                    if MODstdvalue>1:
                        MODstdvalue=-1
                    rowlist.append(MODstdvalue)
                    MODQAvalue = P3QCdata[i][j]
                    QA = dec2bin8(MODQAvalue)
                    QAvalue=0
                    if QA[-7:-5] == "00":  # Second degree polynomials fit
                        QAvalue = 1
                    elif QA[-7:-5] == "01":  # Linear fit
                        QAvalue = 2
                    elif QA[-7:-5] == "10":  # Interpolation between the two nearest dates
                        QAvalue = 3
                    rowlist.append(QAvalue)
            MODISdf.loc[len(MODISdf)] = rowlist
        MODISdf.to_csv("./Data" + s + eachsite + "_HLS_PROBAV300_FAPAR_RMSE_QC.csv")
        print("./Data" + s + eachsite + "_HLS_PROBAV300_FAPAR_RMSE_QC.csv")

        P1000template_dataset = shppath + s + eachsite + "-PROBAV1000.tif"

        MODdata = run.read_img(P1000template_dataset)[2]
        shapelen = MODdata.shape
        Yearlist = []
        Doylist = []
        index = 0
        cloumnnames = ["year", "doy"]
        for i in range(shapelen[0]):
            for j in range(shapelen[1]):
                index = index + 1
                cloumnnames.append("HLS" + str(index))
                cloumnnames.append("PROBAV1000FAPAR" + str(index))
                cloumnnames.append("PROBAV1000STD" + str(index))
                cloumnnames.append("PROBAV1000QA" + str(index))

        MODISdf = pd.DataFrame(columns=(cloumnnames))
        filelist = HLSFAPAR(HLSPROBAV1000PATH + s + eachsite)
        for eachfile in filelist:
            # print(eachfile)
            # MODIS.CA-TP4.HLS.L30.T17TNH.2013196T160546.v2.0.FAPAR.tif
            YEARDOY = os.path.split(eachfile)[1].split(".")[4][:7]
            year = YEARDOY[0:4]
            doy = YEARDOY[4:7]
            month, day = doy2date(int(year), int(doy))
            YEARMODTHDAY = str(year) + str(month).zfill(2) + str(day).zfill(2)
            if day <= 10:
                GEODAY = 10
            elif day > 20:
                GEODAY = 30
            else:
                GEODAY = 20
            GEOMODTHDAY = str(year) + str(month).zfill(2) + str(GEODAY).zfill(2)
            PROBAV1000FAPARfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-FAPAR_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.2.tif"
            if os.path.exists(PROBAV1000FAPARfile)==False:
                PROBAV1000FAPARfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-FAPAR_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.1.tif"
                if os.path.exists(PROBAV1000FAPARfile)==False:
                   continue
                else:
                    PROBAV1000RMSEfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-RMSE_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.1.tif"
                    PROBAV1000QCfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-QFLAG_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.1.tif"
                    if day > 20:
                        for GEODAY in [28, 29, 30, 31]:
                            GEOMODTHDAY = str(year) + str(month).zfill(2) + str(GEODAY).zfill(2)
                            PROBAV1000FAPARfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-FAPAR_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.1.tif"
                            PROBAV1000RMSEfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-RMSE_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.1.tif"
                            PROBAV1000QCfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-QFLAG_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.1.tif"
                            if os.path.exists(PROBAV1000FAPARfile):
                                break
            else:
                PROBAV1000RMSEfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-RMSE_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.2.tif"
                PROBAV1000QCfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-QFLAG_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.2.tif"

                if day > 20:
                    for GEODAY in [28, 29, 30, 31]:
                        GEOMODTHDAY = str(year) + str(month).zfill(2) + str(GEODAY).zfill(2)
                        PROBAV1000FAPARfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-FAPAR_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.2.tif"
                        PROBAV1000RMSEfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-RMSE_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.2.tif"
                        PROBAV1000QCfile = PROBAV1000PATH + s + eachsite + s + eachsite + "_project_" + eachsite + "_c_gls_FAPAR-RT6-QFLAG_" + GEOMODTHDAY + "0000_CUSTOM_PROBAV_V2.0.2.tif"
                        if os.path.exists(PROBAV1000FAPARfile):
                            break

            # HLSP1000data = run.read_img(eachfile)[2]
            # P10FAPARdata = run.read_img(PROBAV1000FAPARfile)[2]/250

            HLSP1000dataall = run.read_img(eachfile)
            im_geotrans, HLSP1000data = HLSP1000dataall[1], HLSP1000dataall[2]
            P10FAPARdataall = run.read_img(PROBAV1000FAPARfile)
            baseim_geotrans, P10FAPARdata = P10FAPARdataall[1], P10FAPARdataall[2] / 250


            P10RMSEdata = run.read_img(PROBAV1000RMSEfile)[2]/250
            P10QCdata = run.read_img(PROBAV1000QCfile)[2]

            shapelen = P10FAPARdata.shape
            # P1000HLS = split(HLSP1000data, shapelen[0], shapelen[1])
            P1000HLS=Stastic_func(baseim_geotrans,P10FAPARdata,im_geotrans,HLSP1000data)
            rowlist = []
            rowlist.append(year)
            tempdate = datetime.datetime.strptime(GEOMODTHDAY, '%Y%m%d')
            doy = tempdate.timetuple().tm_yday
            rowlist.append(doy)
            for i in range(shapelen[0]):
                for j in range(shapelen[1]):
                    # hi = i * shapelen[0] + j
                    tempmean = P1000HLS[i][j]
                    if tempmean==-1:
                        rowlist.extend([-1, -1, -1, -1])
                        continue
                    # shapesum = tempdata.shape
                    # shapesum = shapesum[0] * shapesum[1]
                    # tempdata = tempdata[tempdata > 0]
                    # if len(tempdata) / shapesum < 0.95:
                    #     rowlist.extend([-1, -1, -1, -1])
                    #     continue
                    #
                    # tempmean = np.mean(tempdata)
                    rowlist.append(tempmean)
                    MODvalue = P10FAPARdata[i][j]
                    if MODvalue>1:
                        MODvalue=-1
                    rowlist.append(MODvalue)
                    MODstdvalue = P10RMSEdata[i][j]
                    rowlist.append(MODstdvalue)
                    MODQAvalue = P10QCdata[i][j]
                    QAvalue = 0
                    QA = dec2bin16(MODQAvalue)
                    if QA[-14:-12] == "01":  # Filled with interpolation
                        QAvalue = 3
                    elif QA[-14:-12] == "10":  # Filled with climatology
                        QAvalue = 2
                    elif QA[-14:-12] == "00":  # Driect Retrieve No filled
                        QAvalue = 1
                    rowlist.append(QAvalue)
            MODISdf.loc[len(MODISdf)] = rowlist
        MODISdf.to_csv("./Data" + s + eachsite + "_HLS_PROBAV1000_FAPAR_RMSE_QC.csv")
        print("./Data" + s + eachsite + "_HLS_PROBAV1000_FAPAR_RMSE_QC.csv")

