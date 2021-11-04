# -*- coding: utf-8 -*-
import math
import random
import  copy
import pickle
import pdb

# generate datset
data = [
    [[0.,0.],0.], 
    [[0.,1.],1.], 
    [[1.,0.],1.], 
    [[1.,1.],0.]] * 10000

# pickle file 로 list 저장
with open('data.pickle','wb') as f:
    pickle.dump(data,f)