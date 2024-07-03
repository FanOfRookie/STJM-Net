import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index_1 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward_ori_index_2=[(3,2),(9,2),(5,2),(17,2),(13,2),(18,1),(19,17),(20,18),
                    (14,1),(15,13),(16,14),(4,21),(10,21),(6,21),(7,5),(8,6),(23,7),(22,8),
                    (11,9),(12,10),(25,11),(24,12)]
inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]

class Graph:
    def __init__(self, mode='Partitioning',K=2):
        self.num_node = num_node
        self.self_link = self_link
        self.A = self.get_adjacency_matrix(mode,K)
        self.STJMArray =self.get_transArray()

    def get_adjacency_matrix(self, mode,K):
        edges=[]
        order1 = [ inward_1, self_link,outward_1]
        edges.append(order1)
        if mode == 'Hierarchical':
            A = tools.get_spatial_graph(num_node, tools.get_edgeset(dataset='NTU', CoM=self.CoM)) # L, 3, 25, 25
        elif mode == 'Partitioning':
            if K>=3 or K<0:
                raise ValueError
            if K==2:
                order2 = [inward_2, self_link, outward_2]
                edges.append(order2)
            A = tools.get_spatial_graph(num_node,edges)
        elif mode == 'Disentangled':
            edges=inward_1+self_link+outward_1
            A = tools.get_adjacency_matrix(num_node, edges)
            A=np.stack([tools.k_adjacency(A,i+1) for i in range(K)])
            A=A[np.newaxis,:,:,:]
        else:
            raise ValueError
        return A

    def L1_norm(self,A):
        s=np.sum(A,axis=1,keepdims=True)+ 1e-4
        A=A / s
        return A

    def get_inOut_index(self,A):
        inIndex = []
        outIndex = []
        idenIndex = []
        for i in range(A.shape[0]):
            _inIndex = []
            _outIndex = []
            _idenIndex = []
            for inA, identity,outA in zip(A[i][0] ,A[i][1], A[i][2]):
                __inIndex = []
                __outIndex = []
                __idenIndex = []

                inNonzero = np.nonzero(inA)
                for k in inNonzero[0]:
                    __inIndex.append(k)
                _inIndex.append(__inIndex)

                outNonzero = np.nonzero(outA)
                for k in outNonzero[0]:
                    __outIndex.append(k)
                _outIndex.append(__outIndex)

                idenNonzero = np.nonzero(identity)
                for k in idenNonzero[0]:
                    __idenIndex.append(k)
                _idenIndex.append(__idenIndex)

            inIndex.append(_inIndex)
            outIndex.append(_outIndex)
            idenIndex.append(_idenIndex)
        return idenIndex,inIndex,outIndex

    def get_transArray(self):
        idenIndex, inIndex, outIndex = self.get_inOut_index(self.A)
        STJMArray=[]
        for j in range(self.A.shape[0]):
            _STJMArray=np.zeros([3 * num_node, num_node])
            inCount = np.sum((self.A[j][0]!=0),axis=1)
            selfCount=np.sum((self.A[j][1]!=0),axis=1)
            outCount = np.sum((self.A[j][2]!=0),axis=1)
            for i in range(25):
                if inCount[i] > 0:
                    _STJMArray[3 * i][inIndex[j][i]] = self.A[j][0][i][inIndex[j][i]]
                    _STJMArray[3 * i][inIndex[j][i]] /= inCount[i]
                if selfCount[i] > 0:
                    _STJMArray[3 * i + 1][idenIndex[j][i]] = self.A[j][1][i][idenIndex[j][i]]
                if outCount[i] > 0:
                    _STJMArray[3 * i + 2][outIndex[j][i]] = self.A[j][2][i][outIndex[j][i]]
                    _STJMArray[3 * i + 2][outIndex[j][i]] /= outCount[i]
            _STJMArray=_STJMArray.T
            STJMArray.append(_STJMArray)
        STJMArray=np.stack(STJMArray)
        if self.A.shape[0] == 1:
            STJMArray=STJMArray[np.newaxis,:,:]
        return STJMArray

    # def get_transArray(self):
    #     K=self.A.shape[1]
    #     STJMArray=[]
    #     for i in range(self.A.shape[0]):
    #         _STJMArray=np.zeros([ self.A.shape[1] * num_node,num_node])
    #         for j in range(self.A.shape[1]):
    #             for node in range(self.A.shape[2]):
    #                 _STJMArray[K * node + j,:]=self.A[i,j,node,:]
    #         STJMArray.append(_STJMArray.T)
    #     STJMArray=np.stack(STJMArray)
    #     if STJMArray.shape[0] == 1:
    #         STJMArray=STJMArray[np.newaxis,:,:]
    #     STJMArray=self.L1_norm(STJMArray)
    #     return STJMArray
