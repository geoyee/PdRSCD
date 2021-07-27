import os
import re
import fnmatch
import numpy as np
from PIL import Image
try:
    try:
        from osgeo import gdal
    except ImportError:
        import gdal
    IPT_GDAL = True
except:
    IPT_GDAL = False


def open_tif(geoimg_path, to_np=False):
    '''
        打开tif文件
    '''
    if IPT_GDAL == True:
        geoimg = gdal.Open(geoimg_path)
        if to_np == False:
            return geoimg
        else:
            return tif2array(geoimg), get_geoinfo(geoimg)
    else:
        raise ImportError('can\'t import gdal!')


def tif2array(geoimg):
    if IPT_GDAL == True:
        return geoimg.ReadAsArray().transpose((1, 2, 0))  # 多波段图像默认是[c, h, w]
    else:
        raise ImportError('can\'t import gdal!')


def get_geoinfo(geoimg):
    '''
        获取tif图像的信息，输入为dgal读取的数据
    '''
    if IPT_GDAL == True:
        geoinfo = {
            'xsize': geoimg.RasterXSize,
            'ysize': geoimg.RasterYSize,
            'count': geoimg.RasterCount,
            'proj': geoimg.GetProjection(),
            'geotrans': geoimg.GetGeoTransform()
        }
        return geoinfo
    else:
        raise ImportError('can\'t import gdal!')


def save_tif(img, geoinfo, save_path):
    '''
        保存分割的图像并使其空间信息保持一致
    '''
    if IPT_GDAL == True:
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
        C = img.shape[-1] if len(img.shape) == 3 else 1
        if C == 1:
            dataset.GetRasterBand(1).WriteArray(img)
        else:
            for i_c in range(C):
                dataset.GetRasterBand(i_c + 1).WriteArray(img[:, :, i_c])
        del dataset  # 删除与tif的连接
    else:
        raise ImportError('can\'t import gdal!')


def shp2array(shp_file, tif_file):
    from osgeo import osr

    geoimg = gdal.Open(tif_file)
    trans = geoimg.GetGeoTransform()
    cols = geoimg.RasterXSize
    rows = geoimg.RasterYSize
    mem = gdal.GetDriverByName('MEM')
    mid_ds = mem.Create('', cols, rows, 1, gdal.GDT_Byte)
    mid_ds.SetGeoTransform(trans)
    mid_ds.SetMetadataItem('AREA_OR_POINT', 'Point')
    mid_ds.GetRasterBand(1).WriteArray(np.ones((rows, cols), dtype=np.bool))
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS(geoimg.GetProjection())
    mid_ds.SetProjection(srs.ExportToWkt())
    mask_ds = gdal.Warp('', mid_ds, format='MEM', cutlineDSName=shp_file)
    return mask_ds.ReadAsArray()


def tif2shp(tif_path):
    from osgeo import ogr, osr

    inraster = gdal.Open(tif_path)
    inband = inraster.GetRasterBand(1)
    prj = osr.SpatialReference()  
    prj.ImportFromWkt(inraster.GetProjection())
    outshp = tif_path[:-4] + ".shp"  # 矢量输出文件名
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  # 若文件已经存在，则删除它继续重新做一遍
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
    Poly_layer = Polygon.CreateLayer(tif_path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)
    newField = ogr.FieldDefn('value', ogr.OFTReal)  # 用来存储原始栅格的pixel value
    Poly_layer.CreateField(newField)
    gdal.FPolygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作
    Polygon.SyncToDisk() 
    Polygon = None