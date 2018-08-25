# -*- coding:utf-8 -*-
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time



def load_matrix(filename,num_users,num_items):
    t0 = time.time()
    M = np.zeros((num_users,num_items))
    total = 0.0
    num_zeros = num_users * num_items
    '''如果要对一个列表或者数组既要遍历索引又要遍历元素的时候，可以用enumerate,
    当传入参数为文件时，索引为行号，元素对应的一行内容'''
    for i,line in enumerate(open(filename,'r')):
        user_id,item_id,rating,timestamp = line.strip().split('::')
        user_id = int(user_id) - 1
        item_id = int(item_id) - 1
        rating = float(rating)
        if user_id >= num_users:
            continue
        if item_id >= num_items:
            continue
        if rating != 0:
            M[user_id,item_id] = rating
            total += 1
            num_zeros -= 1
        if i % 1000209 == 0:
            print 'loaded %i counts...' %i
    '''计算稀疏矩阵中的零元素个数和非零元素的个数比例,其实就是论文中的alpha ====> C_ui = 1+alpha*r_ui'''
    alpha = num_zeros / total
    print 'alpha %2.f' %alpha
    M *= alpha  #C_ui = 1+alpha*r_ui
    '''用CompressedSparse Row Fromat　将稀疏矩阵压缩'''
    M = sparse.csr_matrix(M)
    t1 = time.time()
    print 'Finished loading matrix in %f seconds' % (t1 - t0)
    return M    #此处的Ｍ并不是以矩阵的形式存储，所以才有后面的toArray

class ImplicitMF():

    def __init__(self,M,num_factors = 40,num_iterations = 30,reg_param = 0.8):
        self.M = M
        self.num_users = M.shape[0]
        self.num_items = M.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

    def train_model(self):
        '''创建user_vectors,item_vectors,服从Ｎ（0,1）分布，初始化。。。'''
        self.user_vectors = np.random.normal(size=(self.num_users,self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,self.num_factors))
        '''这里用xrange而不用range对于大批量处理数据不需要一次性全部计算不需要开辟大量空间'''
        for i in xrange(self.num_iterations):
            t0 = time.time()
            print 'solving for user vectors...'
            self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
            print 'solving for item vectors...'
            self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))
            t1 = time.time()
            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)
        print self.user_vectors,self.item_vectors

        with open('./solve_vectors','w') as f:
            f.write("user_vectors: " + self.user_vectors + " \n" + "item_vectors: "+ self.item_vectors)
            

    def iteration(self,user,fixed_vecs):
        '''if  user = Ture num_solve = num_users else num_solve = num_items'''
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve,self.num_factors))
        t = time.time()
        for i in xrange(num_solve - 1):
            if user:
                M_i = self.M[i].toarray()#转为数组 每一行
            else:
                # 如果要求item_vec,M_i为M中的第i列的转置　　每一列
                M_i = self.M[:,i].T.toarray()
            ''' 原论文中c_ui=1+alpha*r_ui,但是在计算Y’CuY时为了降低时间复杂度,利用了
                Y'CuY=Y'Y+Y'(Cu-I)Y,由于Cu是对角矩阵,其元素为c_ui，即1+alpha*r_ui。
                所以Cu-I也就是对角元素为alpha*r_ui的对角矩阵'''
            CuI = sparse.diags(M_i,[0])
            pu = M_i.copy()
            # np.where(pu != 0)返回pu中元素不为0的索引，然后将这些元素赋值为1,论文里有介绍ｐ(u)
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI - eye).dot(sparse.csr_matrix(pu).T)
            '''https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html
             求解Ａx = B 的时候x= spsolve(A,B)'''
            xu = spsolve(YTY + YTCuIY + lambda_eye,YTCupu)
            solve_vecs[i] = xu
            if i % 1000 == 0:
                print 'Solved %i vecs in %d seconds' % (i, time.time() - t)
                t = time.time()
        return solve_vecs


if __name__ == '__main__':
    '''ratings.dat 共有1000209条'''
    num_users = 6040
    num_items = 3952
    filename = './ml-100k/ratings.dat'
    M = load_matrix(filename,num_users,num_items)
    # M = [[0, 1, 2, 3, 4], [5, 2, 0, 4, 5], [4, 5, 3, 0, 1], [3, 4, 5, 0, 0]]
    M = sparse.csr_matrix(M)
    print M
    ImplicitMF(M).train_model()



