__author__ = 'Arian'
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import  pandas as pd
from bkt import *
from sklearn.neighbors import KNeighborsClassifier
from KNSS import *
from draw import *


import matlab.engine

eng = matlab.engine.start_matlab()




K_near = 10


instances_number = 595
column_number = 50


img = scipy.io.loadmat('D:\MSC\BD\hws\img\patches.mat')

mg = np.array((img['patches']))
mg_t = mg.T
#print (mg_t.shape)

X = matlab.double(mg_t[0:instances_number,:].tolist())

""" ----------------------------------------------- Actual Data -----------------------------------"""

Prj = np.array(X)
#print (Prj.shape)
#print(pd.DataFrame(np.array(C)))

Prj_Actual= pd.DataFrame(Prj.T,columns = ['p'+str(i) for i in range(Prj.T.shape[1])])
dic_Actual=KNN_SS(K_near , Prj_Actual)

#print "dic_Actual"
#print dic_Actual







""" ----------------------------------------------- Cur article -----------------------------------"""

C,U,R = eng.CUR_article(X ,float(8),0.1,nargout=3)

#print pd.DataFrame(C)
Prj = np.dot(np.array(C),np.array(U) )
#print (Prj.shape)
#print(pd.DataFrame(np.array(C)))

Prj_CUR_article= pd.DataFrame(Prj.T,columns = ['p'+str(i) for i in range(Prj.T.shape[1])])
dic_CUR_article=KNN_SS(K_near , Prj_CUR_article)

#print "dic_CUR_article"
#print dic_CUR_article


""" ----------------------------------------------- Cur -----------------------------------"""


C,U,R = eng.CUR(X , column_number,nargout=3)

#print pd.DataFrame(C)
Prj = np.dot(np.array(C),np.array(U) )
#print (Prj.shape)
#print(pd.DataFrame(np.array(C)))

Prj_CUR= pd.DataFrame(Prj.T,columns = ['p'+str(i) for i in range(Prj.T.shape[1])])
dic_CUR=KNN_SS(K_near , Prj_CUR)

#print "dic_CUR"
#print dic_CUR

#eng.main(nargout=0)

err_CUR = err(dic_CUR , dic_Actual)
print 'Jacard Distance(dic_CUR , dic_Actual)'
print err_CUR

err_CUR_article= err(dic_CUR_article , dic_Actual)

print 'Jacard Distance(dic_CUR_Article , dic_Actual)'
print err_CUR_article

plt.figure(1)
xx,yy= zip(*err_CUR.items())
plt.scatter(range(len(err_CUR.keys())),yy,color='red')

xx,yy= zip(*err_CUR_article.items())
plt.scatter(range(len(err_CUR.keys())),yy)
plt.xlabel('point #')
plt.ylabel('Jacard distance')
plt.title('red:CUR vs Actual//// blue:CUR_article vs Actual '+' K= '+ str(K_near)+' instances_number:= ' + str(instances_number)+ ' dimention :' + str(column_number))
plt.show()











#fig = plt.figure()
##cv2.imshow('Image', img)
##print(img.keys())
#PDS_mg = pd.DataFrame(mg,columns = ['p'+str(i) for i in range(mg.shape[1])])
#"""-------------------------------------------------------  change size of data -------------------------"""
#ten_percent_PDS = PDS_mg.iloc[:,0:594]
##print(mg[:,0].reshape(20,20))
##plt.figure(1)
##for i in range (1,61):
##    plt.subplot(3,20,i)
##    imgplot = plt.imshow(mg[:,i].reshape(20,20))
##
##
##plt.show()
##print [p for p in PDS_mg]
##print list(PDS_mg.columns)
##print bckt_line(np.array([1,2]).T,np.array([0,1]).T,3)
#df_hsh = bcktiz(ten_percent_PDS,10)
##print df_hsh
##print df_hsh.duplicated({False})
##print df_hsh.loc[:,(df_hsh.transpose()).duplicated({False})]
#""" -------------------------------------Q3 p4------------------------------------- """
#kk =10
##
#
##print dups
##print dups['p1171']
##print dups['p2046']
##dd = {'p3':1 ,'p5':2 ,'p9':3, 'p10':4 ,'p2046':5,'p1171':6}
#
#
##
##dd={}
#dups = duplicate_columns(df_hsh,5,4,kk)
##plt.figure(1)
##p_1171 =dups['p1171']
##p_2046 =dups['p2046']
##for i in range(kk):
##    #dd[p_1171[i]]= i
##    plt.subplot(2,kk,i+1)
##    imgplot = plt.imshow(ten_percent_PDS[p_1171[i]].values.reshape(20,20))
##    plt.subplot(2,kk,(i+1)+kk)
##    imgplot = plt.imshow(ten_percent_PDS[p_2046[i]].values.reshape(20,20))
##plt.suptitle("KNN_LSh")
##
##
##plt.figure(2)
##dict_1171 = KNN_Single_Point(kk ,ten_percent_PDS,'p1171' )
##dict_2046 = KNN_Single_Point(kk ,ten_percent_PDS,'p2046' )
###print lst_1171
##for i in range(kk):
##    #dd[p_1171[i]]= i
##    plt.subplot(2,kk,i+1)
##    imgplot = plt.imshow(ten_percent_PDS[dict_1171['p1171'][i][0]].values.reshape(20,20))
##    plt.subplot(2,kk,(i+1)+kk)
##    imgplot = plt.imshow(ten_percent_PDS[dict_2046['p2046'][i][0]].values.reshape(20,20))
##plt.suptitle("KNN")
#
##plt.figure(3)
##plt.subplot(1,2,1)
##imgplot = plt.imshow(ten_percent_PDS['p1171'].values.reshape(20,20))
##plt.title('p1171')
##plt.subplot(2,2,2)
##imgplot = plt.imshow(ten_percent_PDS['p2046'].values.reshape(20,20))
##plt.title('p2046')
#
#""" -----------------------------------------  Q3 p3-----------------"""
#plt.figure(4)
#dict_3_3 = KNN_SS(kk , ten_percent_PDS)
#acc_3_3 = {}
#print dict_3_3
#print dups
##for p in dups.keys():
##    acc_3_3[p] = []
#for p in dups.keys():
#   acc_3_3[p]=(len(list(set(dict_3_3[p])-set(dups[p]))) )
#
#print acc_3_3
#
#plt.scatter([int(x[0][1:]) for x in acc_3_3.items()],[1.0*x[1]/kk for x in acc_3_3.items()])
#plt.xlabel("point id")
#plt.ylabel("diff")
#
#
#
#
#
#
#plt.show()


