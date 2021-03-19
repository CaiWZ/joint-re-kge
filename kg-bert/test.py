# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--out",'-o',
#                         default=None,
#                         type=str,
                        
#                         required=True,
#                         help="The output directory where the model predictions and checkpoints will be written.")
# parser.add_argument("--ind",'-i',
#                         default=None,
#                         type=str,
                        
#                         required=True,
#                         help="The output directory where the model predictions and checkpoints will be written.")                        
# arg=parser.parse_args()
# if arg.out =='f' and arg.ind=='h':
#     print("yesyes")
# else:
#     print('nono')
import sys
from tqdm import trange
import numpy as np
print(sys.version_info)
a=[]
i=0
for i in trange(10):
    if len(a)==0:
        a.append(i)
        i+=1
    else:
        a[0]=np.append(a[0],i)
        i+=1
        print(a[0])
# print(type(a[0]))
