# -*- coding:utf-8 -*-
import numpy as np
import time
import scipy.sparse as sparse



def load_data(filename,num_users,num_items):
    t0 = time.time()
    M = np.zeros((num_users, num_items))
    total = 0.0
    num_zeros = num_users * num_items
    '''如果要对一个列表或者数组既要遍历索引又要遍历元素的时候，可以用enumerate,
    当传入参数为文件时，索引为行号，元素对应的一行内容'''
    # with open(filename,'r') as f:
    #     for i,line in enumerate(f):
    #         tokens = line.strip().split('::')
    #         user_id = int(tokens[0]) - 1
    #         item_id = int(tokens[1]) - 1
    #         rating = int(tokens[2])
    #         timestamp = int(tokens[3])

    for i, line in enumerate(open(filename, 'r')):
        user, item, count,timestamp= line.strip().split('::')
        user = int(user) - 1
        item = int(item) - 1
        count = float(count)
        if user >= num_users:
            continue
        if item >= num_items:
            continue
        if count != 0:
            M[user, item] = count
            total += 1
            num_zeros -= 1
        if i % 1000209 == 0:
            print 'loaded %i counts...' %i
    '''计算稀疏矩阵中的零元素个数和非零元素的个数比例,其实就是论文中的alpha ====> C_ui = 1+alpha*r_ui'''
    alpha = num_zeros / total
    print 'alpha %2.f' % alpha
    # M *= alpha
    '''用CompressedSparse Row Fromat　将稀疏矩阵压缩'''
    M = sparse.csr_matrix(M)
    t1 = time.time()
    print 'Finished loading matrix in %f seconds' % (t1 - t0)
    return M

if __name__ == '__main__':
    '''ratings.dat 共有1000209条'''
    num_users = 6040
    num_items = 3952
    filename = './ml-100k/ratings.dat'
    # M = load_data(filename,num_users,num_items)

    M = [[0,1,2,3,4],[5,2,0,4,5],[4,5,3,0,1],[3,4,5,0,0]]
    M = sparse.csr_matrix(M)
    print M
    print M.shape[0]
    num_solve = M.shape[1]
    for i in xrange(num_solve -1):
        C = M[i].toarray()
        B = M[:,i].T.toarray()# 转为数组
        print C
        print B
        pu = C.copy()
        print pu

