import networkx as nx
import os
import pickle


def find_removed_nodes(origin, removed):
    origin_graph = nx.read_gexf(f'./graph/{origin}')
    removed_graph = nx.read_gexf(f'./tmp/{removed}')
    nodes_left = set(removed_graph.nodes())
    removed_nodes = set()
    for node in origin_graph.nodes():
        if node not in nodes_left:
            removed_nodes.add(node)

    filename = os.path.splitext(origin)[0]
    with open(f'./pickle/{filename}_removed_nodes', 'wb') as f:
        pickle.dump(removed_nodes, f)


def mark_removed_nodes(origin, picklename):
    g = nx.read_gexf(f'./graph/{origin}')
    with open(f'./pickle/{picklename}', 'rb') as f:
        removed_nodes = pickle.load(f)

    for node in g.nodes():
        if node in removed_nodes:
            g.nodes[node]['removed'] = True
        else:
            g.nodes[node]['removed'] = False

    nx.write_gexf(g, f'./graph/{origin[:-5]}_marked.gexf')


if __name__ == '__main__':
    # find_removed_nodes('tweet_1328493390472757253_retweet_net.gexf', \
    #                    'tweet_1328493390472757253_modified_3_1000.gexf')
    mark_removed_nodes('tweet_1328493390472757253_retweet_net.gexf',
                       'tweet_1328493390472757253_retweet_net_removed_nodes')