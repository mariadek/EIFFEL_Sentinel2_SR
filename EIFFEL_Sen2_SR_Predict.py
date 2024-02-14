import argparse
import os
import re
import sys
from osgeo import gdal
from collections import defaultdict
import scipy.ndimage
import numpy as np
from tqdm import tqdm

from tensorflow.keras.layers import Input, Concatenate, Conv2D, Activation, Lambda, Add
from tensorflow.keras.models import Model

import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def s2model(input_shape, num_layers = 32, feature_size = 256):

    input10 = Input(shape = input_shape[0])
    input20 = Input(shape = input_shape[1])

    if len(input_shape) == 3:
        input60 = Input(shape = input_shape[2])
        x = Concatenate(axis=1)([input10, input20, input60])
    else:
        x = Concatenate(axis=1)([input10, input20])


    x = Conv2D(feature_size, (3,3), kernel_initializer = 'he_uniform', activation='relu', padding='same', data_format = 'channels_first')(x)

    for i in range(num_layers):
        x = resBlock(x, feature_size)

    x = Conv2D(input_shape[-1][0], (3,3), kernel_initializer = 'he_uniform', padding='same', data_format = 'channels_first')(x)

    if len(input_shape) ==  3:
        x = Add()([x, input60])
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:
        x = Add()([x, input20])
        model = Model(inputs=[input10, input20], outputs=x)

    return model

def resBlock(x, channels, kernel_size = (3,3), scale=0.1):
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same', data_format = 'channels_first')(x)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same', data_format = 'channels_first')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])

def get_band_short_name(description):
    if ',' in description:
        return description[:description.find(',')]
    if ' ' in description:
        return description[:description.find(' ')]
    return description[:3]

def validate_description(description):
    m = re.match("(.*?), central wavelength (\d+) nm", description)
    if m:
        return m.group(1) + " (" + m.group(2) + " nm)"

    return description

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Perform super-resolution of Sentinel-2 with DSen2",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', action = 'store', dest = 'input_file', help = "An input Sentinel-2 data file")

    args = parser.parse_args()

    output_file = 'SR_' + os.path.split(args.input_file)[-1].split('.')[-2] + '.tiff'
    print(output_file)

    select_bands = 'B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12'

    select_bands = [x for x in re.split(',', select_bands)]

    raster = gdal.Open(args.input_file)

    datasets = raster.GetSubDatasets()
    tenMsets = []
    twentyMsets = []
    sixtyMsets = []
    unknownMsets = []
    for (dsname, dsdesc) in datasets:
        if '10m resolution' in dsdesc:
            tenMsets += [dsname, dsdesc]
        elif '20m resolution' in dsdesc:
            twentyMsets += [dsname, dsdesc]
        elif '60m resolution' in dsdesc:
            sixtyMsets += [dsname, dsdesc]
        else:
            unknownMsets += [dsname, dsdesc]

    validated_10m_bands = []
    validated_10m_indices = []
    validated_20m_bands = []
    validated_20m_indices = []
    validated_60m_bands = []
    validated_60m_indices = []
    validated_descriptions = defaultdict(str)

    sys.stdout.write("Selected 10m bands:")
    ds10 = gdal.Open(tenMsets[0])
    for b in range(0, ds10.RasterCount):
        desc = validate_description(ds10.GetRasterBand(b+1).GetDescription())
        shortname = get_band_short_name(ds10.GetRasterBand(b+1).GetDescription())
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_10m_bands += [shortname]
            validated_10m_indices += [b]
            validated_descriptions[shortname] = desc

    sys.stdout.write("\nSelected 20m bands:")
    ds20 = gdal.Open(twentyMsets[0])
    for b in range(0, ds20.RasterCount):
        desc = validate_description(ds20.GetRasterBand(b+1).GetDescription())
        shortname = get_band_short_name(ds20.GetRasterBand(b+1).GetDescription())
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_20m_bands += [shortname]
            validated_20m_indices += [b]
            validated_descriptions[shortname] = desc

    sys.stdout.write("\nSelected 60m bands:")
    ds60 = gdal.Open(sixtyMsets[0])
    for b in range(0, ds60.RasterCount):
        desc = validate_description(ds60.GetRasterBand(b + 1).GetDescription())
        shortname = get_band_short_name(ds60.GetRasterBand(b + 1).GetDescription())
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_60m_bands += [shortname]
            validated_60m_indices += [b]
            validated_descriptions[shortname] = desc
    sys.stdout.write("\n")

    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(output_file, ds10.RasterXSize, ds10.RasterXSize, 12, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds10.GetGeoTransform())
    outdata.SetProjection(ds10.GetProjection())

    for i, (k, v) in enumerate(validated_descriptions.items()):
        RasterBand = outdata.GetRasterBand(i+1)
        RasterBand.SetDescription(v)

    input_shape_6X = ((4, None, None), (6, None, None), (2, None, None))

    input_shape_2X = ((4, None, None), (6, None, None))

    model_2X = s2model(input_shape_2X, num_layers = 6, feature_size = 128)

    model_2X.load_weights("Best_2X.h5")

    model_6X = s2model(input_shape_6X, num_layers = 6, feature_size = 128)

    model_6X.load_weights("Best_6X_v2.h5")

    chunk_size = 20 # Based on 60 m

    range_i = np.arange(0, ds60.RasterYSize // chunk_size) * chunk_size
    if not(np.mod(ds60.RasterYSize, chunk_size) == 0):
        range_i = np.append(range_i, ds60.RasterYSize - chunk_size)

    for i in tqdm(range_i):
        data10 = ds10.ReadAsArray(xoff = 0, yoff = int(i * 6), xsize = ds10.RasterXSize, ysize = 6 * chunk_size)[validated_10m_indices, :, :]

        data20 = ds20.ReadAsArray(xoff = 0, yoff = int(i * 3), xsize = ds20.RasterXSize, ysize = 3 * chunk_size)[validated_20m_indices, :, :]

        data60 = ds60.ReadAsArray(xoff = 0, yoff = int(i), xsize = ds60.RasterXSize, ysize = chunk_size)[validated_60m_indices, :, :]

        data20_ = np.zeros([data20.shape[0], data20.shape[1]*2, data20.shape[2]*2])
        for k in range(data20.shape[0]):
            data20_[k] = scipy.ndimage.zoom(data20[k], 2, order=1)

        data60_ = np.zeros([data60.shape[0], data60.shape[1]*6, data60.shape[2]*6])
        for l in range(data60.shape[0]):
            data60_[l] = scipy.ndimage.zoom(data60[l], 6, order=1)

        data10 = data10 / 2000.
        data20_ = data20_ / 2000.
        data60_ = data60_ / 2000.

        test_2X = model_2X.predict([data10[np.newaxis,...], data20_[np.newaxis,...]])
        test_6X = model_6X.predict([data10[np.newaxis,...], data20_[np.newaxis,...], data60_[np.newaxis,...]])
        data10  *= 2000
        test_2X *= 2000
        test_6X *= 2000

        test_2X = np.squeeze(test_2X, axis = 0)
        test_6X = np.squeeze(test_6X, axis = 0)

        total = np.concatenate([data10, test_2X, test_6X], axis = 0)

        outdata.WriteArray(total, xoff = 0, yoff = int(i * 6))
        outdata.FlushCache()


    data10 = None
    data20 = None
    data60 = None
    outdata = None
    ds10 = None
    ds20 = None
    ds60 = None
