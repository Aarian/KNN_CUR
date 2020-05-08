__author__ = 'Arian'


def err(dic1, dic2):# map jacard Distance to every point
    """

    :rtype : object
    """
    dic_err = {}
    for key in dic1.keys():
        Jacard_dis = 1 - 1.0*len(list(set(dic1[key]) & set(dic2[key]))) / len(list(set(dic1[key]) | set(dic2[key])))
        dic_err[key] = round(Jacard_dis,2)
    return dic_err