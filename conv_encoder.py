import numpy as np
def conv_encoder(info_seq, transitions):
    info_len = len(info_seq)
    code_len = len(transitions[0,:]) -3

    mem_state = 1
    encoded_seq = np.zeros([code_len,info_len])
    for info_index in range(info_len):
        transitions_index = int((mem_state-1)*2+info_seq[info_index])
        encoded_seq[:,info_index] = transitions[transitions_index,3::]
        mem_state = transitions[transitions_index,1]

    return encoded_seq