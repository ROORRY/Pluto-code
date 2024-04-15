import numpy as np

def polynomial2trellis(generator_polynomial):
    code_len = len(generator_polynomial[:,0])
    mem_num = int(np.floor(np.sqrt(np.max(generator_polynomial))))

    transitions = np.zeros([2**(mem_num+1),3 + code_len],dtype='double')

    count=1
    while True: 
        if np.max(generator_polynomial) < 2**count:
            max_bin_level = count
            break
        count += 1
        
    generator_bs = [np.binary_repr(item,max_bin_level) for items in generator_polynomial.T  for item in items] #dec2bin
    generator_bs = [i[len(i)::-1] for i in generator_bs] #反轉 matalb.fliplr

    temp = [item for items in generator_bs for item in items] #轉成1bit一格
    generator_bs = np.reshape(temp,[np.size(generator_bs),max_bin_level])

    tmp = np.reshape(generator_bs.T,[np.size(generator_bs),1])
    generator_bs_int = np.array([int(i) for i in tmp])
    generator_bin = np.reshape(generator_bs_int,(np.shape(generator_polynomial)+(max_bin_level,)),order="F")

    for memory_state in range(2**mem_num):
        transition_index = memory_state * 2
        transitions[transition_index,0] = memory_state+1
        transitions[transition_index+1,0] = memory_state+1

        temp = [item for items in np.binary_repr(memory_state,mem_num) for item in items]
        temp_int = np.array([int(i) for i in temp])
        bit_memory = np.reshape(temp_int,[1,mem_num],order="F")

        for code_index in range(code_len):
            output_polynomial = np.reshape(generator_bin[code_index,0,:],[1,mem_num+1],order="F")
            recsv_polynomial = np.reshape(generator_bin[code_index,1,:],[1,mem_num+1],order="F")

            # For input 0
            recsv_bit = np.floor(np.mod(np.sum(recsv_polynomial * np.insert(bit_memory,0,0)),2))
            output_bit = np.floor(np.mod(np.sum(output_polynomial * np.insert(bit_memory,0,recsv_bit)),2))
            next_state = np.insert(bit_memory[0,0:-1],0,recsv_bit)

            transitions[transition_index,1] = next_state.dot(1 << np.arange(next_state.shape[-1]-1,-1,-1)) + 1
            transitions[transition_index,2] = 0
            transitions[transition_index,3+code_index] = output_bit

            # For input 1
            recsv_bit = np.floor(np.mod(np.sum(recsv_polynomial * np.insert(bit_memory,0,1)),2))
            output_bit = np.floor(np.mod(np.sum(output_polynomial * np.insert(bit_memory,0,recsv_bit)),2))
            next_state = np.insert(bit_memory[0,0:-1],0,recsv_bit)

            transitions[transition_index+1,1] = next_state.dot(1 << np.arange(next_state.shape[-1]-1,-1,-1)) + 1
            transitions[transition_index+1,2] = 1
            transitions[transition_index+1,3+code_index] = output_bit

    return transitions