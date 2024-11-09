
# 统计每个类别数量
def count_every_sum(a):     
    import collections
    dic = collections.Counter(a)
    for i in dic:
         print(i,dic[i])

