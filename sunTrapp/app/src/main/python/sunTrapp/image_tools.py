import scipy.ndimage

def upsample(data, upsampling):
    data_raw = scipy.ndimage.zoom(data, upsampling, order=0)
    data = scipy.ndimage.zoom(data, upsampling, order=2)
    return data, data_raw