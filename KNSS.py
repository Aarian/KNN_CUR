
import itertools
import numpy as np
from heapq import *



def KNN_SS(K , subspace):# KNN for subspace
    global PDS
    K_near_dic={}
    sub_names =list(subspace.columns)
    for p in sub_names:
        K_near_dic[p] =[]
    for p, q in itertools.combinations(sub_names, 2):
        #if(p != q):
        nrm =np.linalg.norm(subspace[p]-subspace[q])
        K_near_dic[p].append((q,round(nrm,2)))
        K_near_dic[q].append((p,round(nrm,2)))
    for p in subspace:
        pnt_ind = K_near_dic[p]
        K_near_dic[p] = nsmallest(K, pnt_ind, key=lambda x: x[1])
        K_near_dic[p] = [x[0] for x in K_near_dic[p]]
        #idx = np.argpartition(A, K)
        #K_near_dic[p] = A[idx[:K]]
    return K_near_dic

def KNN_Single_Point(K ,subspace, p):# KNN for subspace
    global PDS
    K_near_dic={}
    #for p in subspace:
    K_near_dic[p] =[]
    #for p, q in itertools.combinations(subspace, 2):
    for q in subspace:
        if(p != q):
            nrm =np.linalg.norm(subspace[p]-subspace[q])
            K_near_dic[p].append((q,round(nrm,2)))
        #K_near_dic[q].append((p,round(nrm,2)))
    #for p in subspace:
    pnt_ind = K_near_dic[p]
    K_near_dic[p] = nsmallest(K, pnt_ind, key=lambda x: x[1])
        #idx = np.argpartition(A, K)
        #K_near_dic[p] = A[idx[:K]]
    return K_near_dic


















