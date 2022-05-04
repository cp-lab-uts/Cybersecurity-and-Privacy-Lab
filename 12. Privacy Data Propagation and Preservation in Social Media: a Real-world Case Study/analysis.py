# -*- coding: utf-8 -*-
# @Time    : 2020/6/9 9:40
# @Author  : whm
# @Email   : whm0602@qq.com
# @File    : analysis.py
# @Software: PyCharm


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def analysis_graph(title="analysis data", x_label='degree', y_label='prob'):
    plt.figure()
    plt.title(title)
    plot_degree_distribute('./tmp/Rumor_retweet_net_3.gexf', color='blue',
                           label='rumor', scale=15)

    # plot_degree_distribute('./tmp/Riot_retweet_net_2.gexf', color='black',
    #                        label='Riot_retweet', scale=15)
    #
    # plot_degree_distribute('./tmp/Scale_free_3515.gexf', color='red',
    #                        label='Scale_free', scale=15)
    #
    plot_degree_distribute('./tmp/ER_3515_3321.gexf', color='green',
                           label='ER_graph', scale=15)

    plot_degree_distribute('./tmp/WS_3515_2_0.1.gexf', color='yellow',
                           label='WS_graph', scale=15)

    plot_degree_distribute('./tmp/Riot_cascade.gexf', color='orange',
                           label='news', scale=15)

    # plot_degree_distribute('./tmp/twitter_controled_net_1.gexf', color='green',
    #                        label='Twitter_controled', scale=15)

    plot_degree_distribute('./tmp/KimberlyInfected.gexf', color='red',
                           label='privacy', scale=15)

    plot_degree_distribute('./tmp/BA_3444_1.gexf', color='grey',
                           label='BA_graph', scale=15)

    plot_power_law(4.0, color='yellow', label='gama=4')
    plot_power_law(3.0, color='black', label='gama=3')
    plot_power_law(2.0, color='pink', label='gama=2')
    plot_power_law(1.0, color='green', label='gama=1')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_power_law(gama, color, label):
    X, Y = DataGenerate(gama)
    plt.scatter(X, Y, color=color, label=label)


def DataGenerate(gama):
    X = np.arange(1, 30, 1)  # 0-1，每隔着0.02一个数据  0处取对数,会时负无穷  生成100个数据点
    Y = []
    for i in range(len(X)):
        Y.append(pow(X[i], -gama))  # 得到Y=10.8*x^-0.3+noise

    Y = np.array(Y)
    # X=np.log10(X)  # 对X，Y取双对数
    # Y=np.log10(Y)
    return X, Y


def generate_ER(n, m):
    '''
    Returns a 𝐺𝑛,𝑚 random graph.
    In the 𝐺𝑛,𝑚 model, a graph is chosen uniformly at random from the set of all graphs with 𝑛 nodes and 𝑚 edges.
    This algorithm should be faster than gnm_random_graph() for dense graphs.
    :param n: The number of nodes.
    :param m: The number of edges
    :return:
    '''
    return nx.generators.random_graphs.dense_gnm_random_graph(n, m)


def generate_WS(n, k, p):
    '''
    Return a Watts–Strogatz small-world graph.
    :param n: The number of nodes
    :param k: Each node is joined with its k nearest neighbors in a ring topology.
    :param p: The probability of rewiring each edge
    :return:
    '''
    return nx.generators.random_graphs.watts_strogatz_graph(n, k, p)


def generate_BA(n, m):
    '''
    BA无标度网络
    Returns a random graph according to the Barabási–Albert preferential attachment model.
    A graph of 𝑛 nodes is grown by attaching new nodes each with 𝑚 edges that are preferentially attached to
    existing nodes with high degree.
    :param n: Number of nodes
    :param m: Number of edges to attach from a new node to existing nodes
    :return: G
    '''
    return nx.generators.random_graphs.barabasi_albert_graph(n, m)


def get_max_weakly_connected_component(filename, to_file):
    G = nx.read_gexf(filename)
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(list(largest_cc))
    nx.write_gexf(G, to_file)


def average_clustering(gexf_name):
    G = nx.read_gexf(gexf_name)
    return nx.algorithms.average_clustering(G)


def plot_degree_distribute(gexf_name, color, label, scale=30):
    G = nx.read_gexf(gexf_name)
    degree_histogram = nx.classes.function.degree_histogram(G)
    degree_histogram[1] = degree_histogram[0] + degree_histogram[1]
    degree_histogram.pop(0)
    # print(degree_histogram)
    # print(sum(degree_histogram))
    n = sum(degree_histogram)
    plt.plot([float(k) / n for k in degree_histogram[:scale]], color=color, label=label)


def generate_scale_free_directed_graph(n):
    '''
    Returns a scale-free directed graph.
    :param n: Number of nodes in graph
    :return:
    '''
    G = nx.scale_free_graph(n)
    return G


if __name__ == '__main__':
    # G = generate_ER(3515, 3321)
    # nx.write_gexf(G, './tmp/ER_3515_3321.gexf')
    # G = generate_BA(3444, 1)
    # nx.write_gexf(G, './tmp/BA_3444_1.gexf')
    # G = generate_WS(3515, 4, 0.1)
    # nx.write_gexf(G, './tmp/WS_3515_4_0.1.gexf')
    # G = generate_scale_free_directed_graph(3515)
    # nx.write_gexf(G, './tmp/Scale_free_3515.gexf')
    analysis_graph()
    # get_max_weakly_connected_component('tmp/Rumor_retweet_net3.gexf', 'tmp/tmp.gexf')
    # print(average_clustering('tmp/Rumor_retweet_net_3.gexf'))
    # print(average_clustering('tmp/WS_3444_2_0.02.gexf'))
    # print(average_clustering('tmp/BA_3444_1.gexf'))
    # print(average_clustering('tmp/ER_3444_3340.gexf'))
    # print(average_clustering(('tmp/Riot_retweet_net_2.gexf')))
