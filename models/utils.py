import struct

def dec2binOld(num, places = 8):
    binaryRepStr = "{:0>" + str(places) + "b}"
    return ''.join(binaryRepStr.format(c) for c in struct.pack('!f', num))

def dec2bin(num, places = 16):
    """"
    Function to convert a floating point decimal number to binary number
    """
    whole_list = []
    dec_list = []
    whole, dec = f"{num:.16f}".split('.')
    whole = int(whole)
    dec = int(dec)
    counter = 1

    while (whole / 2 >= 1):
        i = int(whole % 2)
        whole_list.append(i)
        whole /= 2

    decproduct = dec
    while (counter <= places):
        decproduct = decproduct * (10 ** -(len(str(decproduct))))
        decproduct *= 2
        decwhole, decdec = str(decproduct).split('.')
        decwhole = int(decwhole)
        decdec = int(decdec)
        dec_list.append(decwhole)
        decproduct = decdec
        counter += 1

    return dec_list