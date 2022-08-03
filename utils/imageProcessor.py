import gdal

def img_array_to_raster_(file_path, array, num_bands, geotransform, projection, colormap):
    x_pixels = array.shape[0]  # number of pixels in x
    y_pixels = array.shape[1]  # number of pixels in y

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(file_path,
                            x_pixels, y_pixels,
                            num_bands, gdal.GDT_Byte)

    ct = gdal.ColorTable()
    for i in range(len(colormap)):
        ct.SetColorEntry(i, tuple(colormap[i]))

    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).SetRasterColorTable(ct)
    dataset.GetRasterBand(1).SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    dataset.GetRasterBand(1).WriteArray(array)

    dataset.FlushCache()
    del dataset