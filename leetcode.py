import SimpleITK as sitk
import os
import numpy as np
import shutil
import csv























import pandas as pd
import os
l1 = os.listdir("D:/in")
l2 = os.listdir("D:/out")
l3 = []
for i,j in zip(l1,l2):
    str1 = "/home/user/hym2/in/" + i
    str2 = "/home/user/hym2/out/" + j
    print(str1)
    l3.append((str1,str2))
name = ['Image','Mask']
test = pd.DataFrame(columns=name,data=l3)
test.to_csv(r'D:\2.csv')









