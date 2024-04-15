import numpy as np
import conv_encoder as CE

def turbo_encoder(info_seq, transitions, interleaver, select_matrix):
    info_len = len(info_seq)
    encoder_num = 2
    rsccode_len = len(transitions[0,:]) - 3

    if info_len != len(interleaver):
        raise Exception('Information and interleaver length unmatched')
    
    code1 = CE.conv_encoder(info_seq, transitions)
    
    info_seq_itv = np.zeros([len(info_seq),1],dtype='int')
    for index,value in enumerate(interleaver):
        info_seq_itv[index] = info_seq[value]
    code2 = CE.conv_encoder(info_seq_itv,transitions)
    parity_seq_array = np.concatenate([code1[1::,:],code2[1::,:]],axis=0)

    select_seq_array = np.zeros([(rsccode_len-1) * encoder_num,info_len])
    for info_index in  range(info_len):
        select_column = np.mod(info_index+1,len(select_matrix[0,:]))
        select_column = select_column + int(select_column == 0) * len(select_matrix[0,:])
        select_seq_array[:,info_index] = select_matrix[:,select_column-1]

    select_seq_array = np.concatenate([np.ones([1,info_len]),select_seq_array])
    code_seq_array = np.concatenate([np.reshape(info_seq,[1,-1],order='F'),parity_seq_array])
    
    select_seq = np.reshape(select_seq_array,[1,np.size(select_seq_array)],order='F')
    code_seq = np.reshape(code_seq_array,[1,np.size(code_seq_array)],order='F')
    encoded_seq = code_seq[np.where(select_seq == 1)]    
    
    return encoded_seq