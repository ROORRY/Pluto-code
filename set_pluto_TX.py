import adi


def set_Pluto_TX(fc, fs, sn_len, Tx_Gain):
    sdr = adi.Pluto("ip:192.168.2.1")
    sdr.sample_rate = int(fs)

    # Config TX
    sdr.tx_rf_bandwidth = int(fs)
    sdr.tx_lo = int(fc)
    sdr.tx_hardwaregain_chan0 = Tx_Gain

    # Config Rx
    """ 
    sdr.rx_rf_bandwidth = int(fs)
    sdr.rx_lo = int(fc)
    sdr.rx_buffer_size = sn_len*2
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = Rx_Gain
    """
    return sdr
