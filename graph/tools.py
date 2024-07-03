import numpy as np
import torch


def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_graph(num_node, edges):
    In = normalize_digraph(edge2mat(edges[0], num_node))
    Out = normalize_digraph(edge2mat(edges[2], num_node))
    # In = edge2mat(edges[0], num_node)
    I = edge2mat(edges[1], num_node)
    # Out = edge2mat(edges[2], num_node)
    A = np.stack((In, I, Out))
    return A # 3, 25, 25

def get_spatial_graph(num_node,edges):
    A=[]
    for edge in edges:
        A.append(get_graph(num_node,edge))
    A=np.stack(A)
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=True, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

def get_adjacency_matrix(num_nodes,edges ):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A

def get_groups(dataset='NTU', CoM=21):
    groups = []

    if dataset == 'NTU':
        if CoM == 2:
            groups.append([2])
            groups.append([1, 21])
            groups.append([13, 17, 3, 5, 9])
            groups.append([14, 18, 4, 6, 10])
            groups.append([15, 19, 7, 11])
            groups.append([16, 20, 8, 12])
            groups.append([22, 23, 24, 25])

        ## Center of mass : 21
        elif CoM == 21:
            groups.append([21])
            groups.append([2, 3, 5, 9])
            groups.append([4, 6, 10, 1])
            groups.append([7, 11, 13, 17])
            groups.append([8, 12, 14, 18])
            groups.append([22, 23, 24, 25, 15, 19])
            groups.append([16, 20])

        ## Center of Mass : 1
        elif CoM == 1:
            groups.append([1])
            groups.append([2, 13, 17])
            groups.append([14, 18, 21])
            groups.append([3, 5, 9, 15, 19])
            groups.append([4, 6, 10, 16, 20])
            groups.append([7, 11])
            groups.append([8, 12, 22, 23, 24, 25])

        else:
            raise ValueError()

    return groups

def get_edgeset(dataset='NTU', CoM=21):
    groups = get_groups(dataset=dataset, CoM=CoM)

    for i, group in enumerate(groups):
        group = [i - 1 for i in group]
        groups[i] = group

    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []

    for i in range(len(groups) - 1):
        self_link = groups[i] + groups[i + 1]
        self_link = [(i, i) for i in self_link]
        identity.append(self_link)
        forward_g = []
        for j in groups[i]:
            for k in groups[i + 1]:
                forward_g.append((j, k))
        forward_hierarchy.append(forward_g)

        reverse_g = []
        for j in groups[-1 - i]:
            for k in groups[-2 - i]:
                reverse_g.append((j, k))
        reverse_hierarchy.append(reverse_g)

    edges = []
    for i in range(len(groups) - 1):
        edges.append([forward_hierarchy[i], identity[i], reverse_hierarchy[-1 - i]])

    return edges

num_node=25


def L1_norm(A):
    s = np.sum(A, axis=1, keepdims=True) + 1e-4
    A = A / s
    return A

def get_transArray(A,l1_norm=False):
    K = A.shape[1]
    transformer = []
    for i in range(A.shape[0]):
        _transformer = torch.zeros([A.shape[1] * num_node, num_node])
        for j in range(A.shape[1]):
            for node in range(A.shape[2]):
                _transformer[K * node + j, :] = A[i, j, node, :]
        transformer.append(_transformer.T)
    transformer = torch.stack(transformer)
    if transformer.shape[0] == 1:
        transformer = transformer.unsqueeze(0)
    if l1_norm:
        transformer = L1_norm(transformer)
    return transformer