from ctypes import sizeof
from operator import mod
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy import signal
import adi
from SS import SS
from Coding import Coding
from VideoCapture import VideoCapture
from gen_CRC import CRC
from turbo_encoder import turbo_encoder
from polynomial2trellis import polynomial2trellis
import os
import shutil

W = 160
H = 120       # 圖片大小
CH = 1                 # 黑白=1
# 彩色=3

count = 1

Total_bits = H*W*8*CH  # 設定部的資料長度

HS_total = 10           # 設定資料要拆成幾組做傳送

Part_bits = int(Total_bits/HS_total)    # 設定每一組的資料長度
RS_intervel = 2         # 設定RS間隔

N_cpc = 2                 # 設定調變
fft_size = 256          # 設定子載波大小
CRC_Type = 241          # CRC的種類
CRC_SIZE = 24           # CRC的大小
code_rate = 1/3

Rx_gain = 30           # 設定feedback_rx的功率大小
Tx_gain = 0          # 設定tx的功率大小

fc = 3500e6                # 設定data的中心頻率
fc_HS = 2000e6      # 設定feedback的中心頻率
fs = 60e6                     # 設定採樣頻率

total_path = "Picture"
if os.path.exists(total_path):
    shutil.rmtree(total_path)
os.mkdir(total_path)
#################      data        #########################
while(True):

    #test_data = np.random.randint(0, 2, [1, Total_bits], dtype='int64')

    path = total_path + "/" + str(count)
    os.makedirs(path)
    '''
    VideoCapture(W, H, CH, path)
    filename = "BIN_DATA.txt"  # 讀檔
    fp = open(filename, "r")
    RSQ_line = fp.read()
    fp.close()
    RSQ = np.array(RSQ_line.split(', '), dtype='int64')
    test_data = np.array(RSQ)
    '''
    filename = "1_gray_120160.txt"  # 讀檔
    fp = open(filename, "r")
    RSQ_line = fp.read()
    fp.close()
    RSQ = np.array(RSQ_line.split(', '), dtype='int64')
    test_data = np.array(RSQ)
    #################      Index        #########################
    # 注意，這裡配合matlab，index從1開始算
    Inx_carrier = np.arange(1, 64+1)
    Inx_carrier = np.insert(Inx_carrier, np.size(
        Inx_carrier), range(256-64+1, 257))

    Inx_carrier_sync = np.arange(1, 63+1)
    Inx_carrier_sync = np.insert(Inx_carrier_sync, np.size(
        Inx_carrier_sync), range(256-63, 257))

    test_data = np.reshape(test_data, [Part_bits, -1], order='F').T # 將資料分成n組
    test_data_CRC_num = Part_bits + CRC_SIZE
    test_data_encoding_num = test_data_CRC_num/code_rate
    RS_num = np.floor(test_data_encoding_num/N_cpc /
                      np.size(Inx_carrier)/(RS_intervel-1)) + 1  # RS的數量
    Num_Data_Bit = RS_num*N_cpc*np.size(Inx_carrier)*(RS_intervel-1)

    while np.mod(Num_Data_Bit, 1/code_rate) != 0 or np.mod(Num_Data_Bit-test_data_encoding_num, 1/code_rate) != 0:
        RS_num = RS_num + 1
        Num_Data_Bit = RS_num*N_cpc*np.size(Inx_carrier)*(RS_intervel-1)

    a = test_data[:, int(
        np.size(test_data, 1)-(Num_Data_Bit-test_data_encoding_num)*code_rate):np.size(test_data, 1)]
    Data_bit = np.concatenate([test_data, a], axis=1)
    RS_end = RS_num*RS_intervel + 2  # 最後一個symbol的位置

    Inx_RS = np.arange(2, RS_end+1, RS_intervel, dtype="int")
    Inx_data = np.arange(3, Inx_RS[-1], dtype="int")
    Inx_sync = np.array([1])    # 將同步位元放在第一個symbol
    for i in range(0, len(Inx_RS)-1):
        Inx_temp = np.where(Inx_data == Inx_RS[i])
        Inx_data = np.delete(Inx_data, Inx_temp)

    Total_len = np.size(Inx_data) + np.size(Inx_RS) + np.size(Inx_sync)
    CP_length = 32

    tx_data_freq = np.zeros([fft_size, Total_len], dtype="complex")
    #############################################################
    #################       PSS         #########################
    N1 = 0
    N2 = 0
    [CellID, PSS, SSS] = SS(N1, N2)
    tx_data_freq[Inx_carrier_sync-1, Inx_sync-1] = PSS
    #############################################################
    #################      CSIRS        #########################

    def CSIRS(n, l, Num_Used_Carrier):
        N = 14
        nID = 0
        CSI_C_init = ((2**10)*(N*n+l+1)*(2*nID+1)+nID) % 2**31

        x1 = np.array(np.zeros(10031))
      
        x1[0] = 1
        x = 31
        for m in range(10000):
            x1[m+x] = mod(x1[m+3]+x1[m], 2)

        xx = np.array(np.zeros(31))
        cont = 0

        for i in range(30, -1, -1):
            if CSI_C_init == 0:
                continue
            if np.floor(np.log2(CSI_C_init)) == i :
                xx[cont] = i
                CSI_C_init = CSI_C_init - 2**(xx[cont])
                cont += 1

        xx = xx[0:cont]
        x2 = np.zeros(10031)

        for i in range(len(xx)):
            x2[int(xx[i])] = 1
        x = 31

        for m in range(10000):
            x2[m+x] = mod((x2[m+3]+x2[m+2]+x2[m+1]+x2[m]), 2)
        C = np.zeros(Num_Used_Carrier*2+1)

        for m in range(np.size(C)):
            C[m] = mod(x1[m+1600]+x2[m+1600], 2)
        r = np.zeros(Num_Used_Carrier, dtype="complex")

        for m in range(Num_Used_Carrier):
            r[m] = ((1/np.sqrt(2))*(1-2*C[2*m+1])) + \
                1j*(1/np.sqrt(2))*(1-2*C[2*m+2])

        return r

    for k in range(np.size(Inx_RS)):
        c = Inx_RS[k]
        n = mod(np.floor(c/14)+1, 10)
        l = mod(c, 14)
        RS_Seq = CSIRS(n, l, np.size(Inx_carrier))
        tx_data_freq[Inx_carrier-1, Inx_RS[k]-1] = (RS_Seq)

    #############################################################
    #################       Data        #########################
    final_data = np.zeros(np.shape(test_data), dtype='int64')

    for HS_index in range(HS_total):
        #############################################################
        #####################      CRC        #######################
        Data_Bit = CRC(Data_bit[HS_index, :], CRC_Type)  # add CRC bits
        #############################################################
        #################      Turbo code        ####################
        # Variables
        info_len = np.size(Data_Bit)  # Frame length
        max_iteration = 20  # Turbo decoder iteration upper bound
        Ec_No_dB = 100000  # Channel SNR

        # Generate RSC encoders
        transitions = polynomial2trellis(
            np.array([[1, 1], [5, 7]], dtype='int'))

        # Generate information sequence
        info_seq = Data_Bit

        # Generate interleaver and de-interleaver
        block = int(np.floor(np.sqrt(info_len))+1)
        interleaver = np.append(np.arange(info_len), -
                                np.ones([1, block**2-info_len], dtype='int'))
        interleaver = np.reshape(
            np.rot90(np.reshape(interleaver, [block, block]), 2), [1, -1], order='F')
        temp_index = np.where(interleaver == -1)
        interleaver = np.delete(interleaver, temp_index)

        # Generate select matrix
        select_matrix = np.array([[1, 1], [1, 1]], dtype='int')

        # Conduct channel coding

        encoded_seq = turbo_encoder(
            info_seq, transitions, interleaver, select_matrix)

        #encoded_seq = info_seq
        #############################################################
        #################      Modulation        ####################
        tx_data_mod_bef_reshape = Coding.modulation(encoded_seq, N_cpc)
        tx_data_mod = np.reshape(tx_data_mod_bef_reshape,
                                 (np.size(Inx_carrier), -1), order="F")
        for i in range(np.size(Inx_data)):
            tx_data_freq[Inx_carrier-1, Inx_data[i]-1] = tx_data_mod[:, i]

        #############################################################
        ##################       CP          ########################
        tx_data = (np.fft.ifft(tx_data_freq.T)*np.sqrt(fft_size)).T
        tx_data_CP = np.zeros([fft_size+CP_length, Total_len], dtype="complex")
        tx_data_CP[0:CP_length, :] = tx_data[(len(tx_data) - CP_length):, :]
        tx_data_CP[CP_length:, :] = tx_data
        #############################################################
        #################     Feedback        #######################
        if HS_index != 0:
            if mod(HS_index, 2) == 1:
                feedback_signal = np.array(
                    PSS, dtype="complex")  # feedback傳的資料
                tx_data_freq[Inx_carrier_sync-1, Inx_sync -
                             1] = PSS     # 下組資料用的同步訊號，用來判斷是否已傳下一組
            else:
                feedback_signal = np.array(SSS, dtype="complex")
                tx_data_freq[Inx_carrier_sync-1, Inx_sync - 1] = SSS

            # Config Rx(一台要註解掉)
            sdr.rx_lo = int(fc_HS)
            sdr.rx_buffer_size = len(feedback_signal)
            sdr.gain_control_mode_chan0 = 'manual'
            sdr.rx_hardwaregain_chan0 = Rx_gain

            # 判斷是否收到feedback
            while True:
                # Receive samples
                rx_HS = sdr.rx()
                a = signal.correlate(rx_HS, feedback_signal, "full")

                a_abs = abs(a)
                a_abs = sorted(a_abs)
                max1 = a_abs[-1]
                max2 = a_abs[-10]

                if(max1 > 3.5*max2):
                    print("已收到feedback")
                    sdr.tx_destroy_buffer()
                    break
        else:
            tx_data_freq[Inx_carrier_sync-1, Inx_sync - 1] = SSS

        #############################################################
        #################     PlutoSDR        #######################
        tx_data_seq = np.reshape(tx_data_CP.T, [-1])

        sdr = adi.Pluto("ip:192.168.2.1")
        sdr.sample_rate = int(fs)

        # Config Tx
        # filter cutoff, just set it to the same as sample rate
        sdr.tx_rf_bandwidth = int(fs)
        sdr.tx_lo = int(fc)
        sdr.tx_hardwaregain_chan0 = Tx_gain

        sdr.tx_cyclic_buffer = True
        samples = tx_data_seq
        samples *= 2**12
        sdr.tx(samples)
        print("已傳送第 ", HS_index+1, " 組訊號")
        
    count+=1

#############################################################
input("STOP")
