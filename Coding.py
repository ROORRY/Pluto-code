import numpy as np


class Coding:

    # BPSK QPSK 16-QAM 64-QAM 1024-QAM

    def modulation(bk, N_cpc):
        if N_cpc == 1:  # BPSK
            tmp = -bk*2+1
            xk = (tmp+1j*tmp)
            xk = xk/(2**0.5)

        elif N_cpc == 2:  # QPSK
            xk = np.zeros(int(len(bk)/N_cpc), dtype=complex)
            cnt_bk = 0
            for n in range(0, len(xk)):
                xk[n] = (-bk[cnt_bk]*2+1)
                cnt_bk += 1
                xk[n] = xk[n]+1j*(-bk[cnt_bk]*2+1)
                cnt_bk += 1
            xk = xk/(2**0.5)

        elif N_cpc == 4:  # 16-QAM
            xk = np.zeros(int(len(bk)/N_cpc), dtype=complex)
            cnt_bk = 0
            for n in range(0, len(xk)):
                # I-channel
                if bk[cnt_bk] == 1 and bk[cnt_bk+2] == 1:
                    xk[n] = xk[n]-3
                elif bk[cnt_bk] == 1 and bk[cnt_bk+2] == 0:
                    xk[n] = xk[n]-1
                elif bk[cnt_bk] == 0 and bk[cnt_bk+2] == 0:
                    xk[n] = xk[n]+1
                else:
                    xk[n] = xk[n]+3
                # Q-channel
                if bk[cnt_bk+1] == 1 and bk[cnt_bk+3] == 1:
                    xk[n] = xk[n]-1j*3
                elif bk[cnt_bk+1] == 1 and bk[cnt_bk+3] == 0:
                    xk[n] = xk[n]-1j
                elif bk[cnt_bk+1] == 0 and bk[cnt_bk+3] == 0:
                    xk[n] = xk[n]+1j
                else:
                    xk[n] = xk[n]+1j*3
                cnt_bk = cnt_bk+N_cpc
            xk = xk/(10**0.5)

        elif N_cpc == 6:  # 64-QAM
            xk = np.zeros(int(len(bk)/N_cpc), dtype=complex)
            cnt_bk = 0
            for n in range(0, len(xk)):
                # I-channel
                if bk[cnt_bk] == 1 and bk[cnt_bk+2] == 1 and bk[cnt_bk+4] == 1:
                    xk[n] = xk[n]-7
                elif bk[cnt_bk] == 1 and bk[cnt_bk+2] == 1 and bk[cnt_bk+4] == 0:
                    xk[n] = xk[n]-5
                elif bk[cnt_bk] == 1 and bk[cnt_bk+2] == 0 and bk[cnt_bk+4] == 0:
                    xk[n] = xk[n]-3
                elif bk[cnt_bk] == 1 and bk[cnt_bk+2] == 0 and bk[cnt_bk+4] == 1:
                    xk[n] = xk[n]-1
                elif bk[cnt_bk] == 0 and bk[cnt_bk+2] == 0 and bk[cnt_bk+4] == 1:
                    xk[n] = xk[n]+1
                elif bk[cnt_bk] == 0 and bk[cnt_bk+2] == 0 and bk[cnt_bk+4] == 0:
                    xk[n] = xk[n]+3
                elif bk[cnt_bk] == 0 and bk[cnt_bk+2] == 1 and bk[cnt_bk+4] == 0:
                    xk[n] = xk[n]+5
                else:
                    xk[n] = xk[n]+7
                # Q-channel
                # I-channel
                if bk[cnt_bk+1] == 1 and bk[cnt_bk+3] == 1 and bk[cnt_bk+5] == 1:
                    xk[n] = xk[n]-1j*7
                elif bk[cnt_bk+1] == 1 and bk[cnt_bk+3] == 1 and bk[cnt_bk+5] == 0:
                    xk[n] = xk[n]-1j*5
                elif bk[cnt_bk+1] == 1 and bk[cnt_bk+3] == 0 and bk[cnt_bk+5] == 0:
                    xk[n] = xk[n]-1j*3
                elif bk[cnt_bk+1] == 1 and bk[cnt_bk+3] == 0 and bk[cnt_bk+5] == 1:
                    xk[n] = xk[n]-1j*1
                elif bk[cnt_bk+1] == 0 and bk[cnt_bk+3] == 0 and bk[cnt_bk+5] == 1:
                    xk[n] = xk[n]+1j*1
                elif bk[cnt_bk+1] == 0 and bk[cnt_bk+3] == 0 and bk[cnt_bk+5] == 0:
                    xk[n] = xk[n]+1j*3
                elif bk[cnt_bk+1] == 0 and bk[cnt_bk+3] == 1 and bk[cnt_bk+5] == 0:
                    xk[n] = xk[n]+1j*5
                else:
                    xk[n] = xk[n]+1j*7
                cnt_bk = cnt_bk+N_cpc
            xk = xk/(42**0.5)

        elif N_cpc == 8:  # 256-QAM
            xk = np.zeros(int(len(bk)/N_cpc), dtype=complex)
            cnt_bk = 0
            for n in range(0, len(xk)):
                bit_0 = (1-2*bk[cnt_bk])
                bit_2 = (1-2*bk[cnt_bk+2])
                bit_4 = (1-2*bk[cnt_bk+4])
                bit_6 = (1-2*bk[cnt_bk+6])
                xk_real = bit_0*(8-bit_2*(4-bit_4*(2-bit_6)))

                bit_1 = (1-2*bk[cnt_bk+1])
                bit_3 = (1-2*bk[cnt_bk+3])
                bit_5 = (1-2*bk[cnt_bk+5])
                bit_7 = (1-2*bk[cnt_bk+7])
                xk_imag = 1j*bit_1*(8-bit_3*(4-bit_5*(2-bit_7)))
                cnt_bk = cnt_bk+8
                xk[n] = xk_real+xk_imag
            xk = xk/(170**0.5)

        elif N_cpc == 10:  # 1024-QAM
            xk = np.zeros(int(len(bk)/N_cpc), dtype=complex)
            cnt_bk = 0
            for n in range(0, len(xk)):
                bit_0 = (1-2*bk[cnt_bk])
                bit_2 = (1-2*bk[cnt_bk+2])
                bit_4 = (1-2*bk[cnt_bk+4])
                bit_6 = (1-2*bk[cnt_bk+6])
                bit_8 = (1-2*bk[cnt_bk+8])
                xk_real = bit_0*(16-bit_2*(8-bit_4*(4-bit_6*(2-bit_8))))

                bit_1 = (1-2*bk[cnt_bk+1])
                bit_3 = (1-2*bk[cnt_bk+3])
                bit_5 = (1-2*bk[cnt_bk+5])
                bit_7 = (1-2*bk[cnt_bk+7])
                bit_9 = (1-2*bk[cnt_bk+9])
                xk_imag = 1j*bit_1*(16-bit_3*(8-bit_5*(4-bit_7*(2-bit_9))))
                cnt_bk = cnt_bk+10
                xk[n] = xk_real+xk_imag
            xk = xk/(682**0.5)

        return xk

    # 現只有BPSK、QPSK
    def demodulation(xk, N_cpc):
        yk = np.zeros(len(xk))
        if N_cpc == 1:  # BPSK
            re_xk = np.real(xk)
            im_xk = np.imag(xk)
            for n in range(0, len[xk] - 1):
                if im_xk[n] >= -re_xk[n]:
                    yk[n] = 1+1j
                else:
                    yk[n] = -1-1j

        elif N_cpc == 2:  # QPSK
            re_xk = np.real(xk)
            im_xk = np.imag(xk)
            re_yk = (re_xk >= 0) * 2.0 - 1
            im_yk = (im_xk >= 0) * 2.0 - 1
            yk = re_yk + 1j*im_yk
        return yk
