import sys
import os

filename = sys.argv[1]
filename = os.getcwd()+'/'+filename

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
    
print("Linecount: {}".format(file_len(filename)))