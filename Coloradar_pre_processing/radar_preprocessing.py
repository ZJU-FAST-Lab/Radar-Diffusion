import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict as edict
import os
import pdb
import radardsp


def RAmap(radar_adc_data,radar_config, tx_array, rx_array):
  
    ntx, nrx, nc, ns = radar_adc_data.shape

    radar_adc_data *= np.blackman(ns).reshape(1, 1, 1, -1)

    rfft = np.fft.fft(radar_adc_data, radar_config.range_fftsize, -1)    

    dfft = np.fft.fft(rfft, radar_config.doppler_fftsize, -2)
    dfft = np.fft.fftshift(dfft, -2)
    vcomp = radardsp.velocity_compensation(ntx, radar_config.doppler_fftsize)
    dfft *= vcomp
    
    # print("tx_array
    # ", tx_array)
    # print("rx_array", rx_array)
    _dfft = radardsp.virtual_array(dfft, tx_array, rx_array)

    afft = np.fft.fft(_dfft, radar_config.angle_fftsize, 1)
    afft = np.fft.fftshift(afft, 1)

    # Elevation esitamtion
    efft = np.fft.fft(afft, radar_config.angle_fftsize, 0)
    efft = np.fft.fftshift(efft, 0)

    efft[:, :, :, 0:int(efft.shape[-1] * radar_config.crop_low)] = 0
    efft[:, :, :, -int(efft.shape[-1] * radar_config.crop_high):] = 0

    FFT_power = np.abs(efft) ** 2

    FFT_power = np.sum(FFT_power, (0, 2))
    # print("FFT_power", FFT_power.shape)

    noise = np.quantile(FFT_power, 0.30, (0, 1))
    FFT_power /= noise
    # print("dpcl", dpcl)
    #zrb warning
    dpcl = 10 * np.log10(FFT_power + 1)

    dpcl_trans = np.transpose(dpcl, (1, 0))

    return dpcl_trans

def adc2pcd_coloradar(radar_adc_data,radar_config, tx_array, rx_array):
    ntx, nrx, nc, ns = radar_adc_data.shape
    radar_adc_data *= np.blackman(ns).reshape(1, 1, 1, -1)

    rfft = np.fft.fft(radar_adc_data, radar_config.range_fftsize, -1)

    rfft[:, :, :, 0:int(rfft.shape[-1] * radar_config.crop_low)] = 0
    rfft[:, :, :, -int(rfft.shape[-1] * radar_config.crop_high):] = 0
    

    dfft = np.fft.fft(rfft, radar_config.doppler_fftsize, -2)
    dfft = np.fft.fftshift(dfft, -2)

    # print("dfft", dfft.shape)
    mimo_dfft = dfft.reshape(ntx * nrx, radar_config.doppler_fftsize, radar_config.range_fftsize)
    mimo_dfft = np.sum(np.abs(mimo_dfft) ** 2, 0)

    # print("dfft", dfft.shape)
    va = radardsp.virtual_array(dfft, tx_array, rx_array)
    ne, na, nv, nr = va.shape
    FFT_power = np.abs(va) ** 2
    FFT_power = np.sum(FFT_power, (0, 1)) / (ne * na)
    noise = np.quantile(FFT_power, 0.10, (0, 1))
    FFT_power /= noise
    # print("FFT_power", FFT_power.size)
    sig_integrate = 10 * np.log10(FFT_power + 1)

    min_value = np.min(sig_integrate)
    max_value = np.max(sig_integrate)

    _, detections = radardsp.nq_cfar_2d(
        mimo_dfft,
        radar_config.RD_OS_CFAR_WS_COARSE,
        radar_config.RD_OS_CFAR_GS_COARSE,
        radar_config.RD_OS_CFAR_K_COARSE,
        radar_config.RD_OS_CFAR_TOS_COARSE,
    )


    va_nel, va_naz, va_nc, va_ns = va.shape


    Ne, Na, Nc, Ns = radardsp._get_fft_size(*va.shape)

    print(f"va_nel: {va_nel}, va_naz: {va_naz}, va_nc: {va_nc}, va_ns: {va_ns}")
    print(f"Ne: {Ne}, Na: {Na}, Nc: {Nc}, Ns: {Ns}")

    va = np.pad(
        va,
        (
            (0, Ne - va_nel), (0, Na - va_naz),
            (0, Nc - va_nc), (0, Ns - va_ns)
        ),
        "constant",
        constant_values=((0, 0), (0, 0), (0, 0), (0, 0))
    )

    rbins, vbins, abins, ebins = radardsp._get_bins(Ns, Nc, Na, Ne, radar_config)

    # print("abins", abins.shape)
    # print("ebins", ebins.shape)
    pcl = []
    # print("detections", len(detections))
    cnt = 0
    for idx, obj in enumerate(detections):
        obj.range = rbins[obj.ridx]
        obj.velocity = vbins[obj.vidx]
        afft = np.fft.fft(va[:, :, obj.vidx, obj.ridx], radar_config.angle_fftsize, 1)
        afft = np.fft.fftshift(afft, 1)
        mask = radardsp.os_cfar(
            np.abs(np.sum(afft, 0)).reshape(-1),
            radar_config.AZ_OS_CFAR_WS_COARSE,
            radar_config.AZ_OS_CFAR_GS_COARSE,
            radar_config.AZ_OS_CFAR_TOS_COARSE,
        )
        _az = np.argwhere(mask == 1).reshape(-1)
        # print("_az", _az.shape)
        for _t in _az:
            cnt = cnt + 1
            efft = np.fft.fft(afft[:, _t], radar_config.elevation_fftsize, 0)
            efft = np.fft.fftshift(efft, 0)
            _el = np.argmax(efft)
            obj.az = abins[_t]
            obj.el = ebins[_el]

            pcl.append(np.array([
                obj.az,                     # 0 Azimnuth
                obj.range,                  # 1 Range
                obj.el,                     # 2 Elevation
                obj.velocity,               # 3 Velocity
                10 * np.log10(obj.snr),     # 4 SNR,
                obj.ridx,                   # 5 range_index
                obj.vidx,                   # 6 doppler_index
            ]))

            
    # avoiding empty pcl
    if len(pcl) == 0:
      pcl.append(np.array([
          0,                     # 0 Azimnuth
          0,                   # 1 Range
          0,                      # 2 Elevation
          0,               # 3 Velocity
          0,     # 4 SNR,
          0,                   # 5 range_index
          0,                   # 6 doppler_index
      ]))

    pcl = np.array(pcl)        
    # print("cnt", cnt)
    print("pcl", pcl.shape)
    # Remove very close range
    # pcl = pcl[pcl[:, 1] >= 1.5]
    # max_range = (3e8 * radar_config.chirpRampTime * radar_config.Fs)
    # pcl = pcl[pcl[:, 1] < (0.95 * max_range)]
    # pcl = pcl[pcl[:, 4] > np.max(pcl[:, 4]) * 0.4]
    

    pcld = np.zeros(pcl.shape)
    pcld[:, 0] = pcl[:, 1] * np.cos(pcl[:, 2]) * np.sin(pcl[:, 0])
    pcld[:, 1] = pcl[:, 1] * np.cos(pcl[:, 2]) * np.cos(pcl[:, 0])
    pcld[:, 2] = pcl[:, 1] * np.sin(pcl[:, 2])
    pcld[:, 3:] = pcl[:, 3:]
  

    # pcld = pcld[pcl[:, 1] >= 0]

    return np.array(pcld), dfft, sig_integrate