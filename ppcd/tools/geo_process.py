import gdal
import numpy as np


def open_tif(geoimg_path, to_np=False):
    '''
        打开tif文件
    '''
    geoimg = gdal.Open(geoimg_path)
    if to_np == False:
        return geoimg
    else:
        return geoimg.ReadAsArray().transpose((1, 2, 0))  # 多波段图像默认是[c, h, w]


def get_geoinfo(geoimg):
    '''
        获取tif图像的信息，输入为dgal读取的数据
    '''
    geoinfo = {
        'xsize': geoimg.RasterXSize,
        'ysize': geoimg.RasterYSize,
        'count': geoimg.RasterCount,
        'proj': geoimg.GetProjection(),
        'geotrans': geoimg.GetGeoTransform()
    }
    return geoinfo


def save_tif(img, geoinfo, save_path):
    '''
        保存分割的图像并使其空间信息保持一致
    '''
    driver = gdal.GetDriverByName('GTiff')
    datatype = gdal.GDT_Byte
    dataset = driver.Create(
        save_path, 
        geoinfo['xsize'], 
        geoinfo['ysize'], 
        geoinfo['count'], 
        datatype)
    dataset.SetProjection(geoinfo['proj'])  # 写入投影
    dataset.SetGeoTransform(geoinfo['geotrans'])  # 写入仿射变换参数
    dataset.GetRasterBand(1).WriteArray(img)
    del dataset  # 删除与tif的连接