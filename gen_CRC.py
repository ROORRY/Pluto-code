import numpy as np

def mod_2(data,P):
    if(data[0] == 1):
        dividend = data[0:len(P)]
        down_index = len(P)
    else:
        dividend = data[1:len(P)+1]
        down_index = len(P) + 1

    while down_index < len(data) +1 and len(dividend) == len(P):
        remainder = np.zeros([len(P)],dtype='int')
        for j in range(len(P)):
            remainder[j] = int(dividend[j]) ^ P[j]       
        
        temp = np.where(remainder == 1)[0][0]
        if len(data)- down_index < temp:
            remainder = np.delete(remainder,np.arange(len(data)- down_index+1))
            remainder = np.append(remainder,data[down_index:len(data)])
            break
        remainder = np.delete(remainder,np.arange(temp))
        remainder = np.append(remainder,data[down_index:down_index+temp])
        down_index += temp
        dividend = remainder
        

    
    return remainder
        
def CRC(data,CRC_Type):
    
    if CRC_Type == 11:
        P = [1,1,1,0,0,0,1,0,0,0,0,1]
    elif CRC_Type == 16:
        P = [1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1]
    elif CRC_Type == 241: #CRC24A
        P = [1,1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1]
    elif CRC_Type == 242: #CRC24B
        P = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1]
    elif CRC_Type == 243: #CRC24C
        P = [1,1,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,1]


    rdata = np.append(data,np.zeros([1,len(P)-1]))
    remainder = mod_2(rdata,P)

    output = np.concatenate([data,remainder])

    return output



