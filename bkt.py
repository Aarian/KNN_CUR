__author__ = 'Arian'
import numpy as np
import  pandas as pd
from copy import deepcopy
def bckt_line(pnt,v,width):
      v_n = v/np.sum(v**2)

      return int(np.dot(v_n.T,pnt)/width)
def bcktiz(PDS,width):
    bkt_df = pd.DataFrame(index =['v'+str(i) for i in range(20)],columns =PDS.columns )
    for hsh in range(20):
        bkt_lst = []
        v = np.random.uniform(low=0,high = 5,size =(PDS.shape[0],1))
        for p in PDS :
            #print v
            bkt_number = bckt_line(PDS[p],v,width)
            bkt_lst.append(bkt_number)
        bkt_df.loc['v'+str(hsh)] = bkt_lst
    return bkt_df



def duplicate_columns(PDS_hsh,r,b,K):
    frame = (PDS_hsh)
    dups = []
    sim_dic ={}
    lst = list(frame)
    n= len(lst)
    or_lst =[]
    for i in range(n):
        sim_dic[lst[i]] = []

    for i in range(n):
        for j in range(i+1,n):
            or_lst=[]
            if len(sim_dic[lst[i]]) == K :
                break
            for bb in range(b):
                if ((frame.iloc[bb*r:(bb+1)*r,i] == frame.iloc[bb*r:(bb+1)*r,j]).all()):
                    sim_dic[lst[i]].append(lst[j])
                    break
    return sim_dic # dups