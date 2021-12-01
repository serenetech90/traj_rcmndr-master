''' class online graph is to construct and adapt underlying graph as soon as new step is take by pedestrian '''

import networkx as nx
import numpy as np
import tensorflow as tf
import torch
# import torch.cuda
# import criterion as cr

batch_size = 4

class online_graph():

    def reset_graph(self,framenum):
        del self.nodes
        del self.edges
        self.onlineGraph.delGraph(framenum)

        self.nodes = [{}]
        self.edges = [[]]

    def __init__(self, args):
        self.diff = args.obs_len
        self.pred_diff = args.pred_len
        self.batch_size = args.batch_size  # 1
        self.ratio = self.pred_diff / self.diff
        # self.seq_length = args.seq_length # 1
        self.nodes = [{}]
        self.edges = [{}]
        self.onlineGraph = Graph(self.diff, self.pred_diff)

  
    def ConstructGraph(self, current_batch, vislets, future_traj, framenum, stateful=True, valid=False):
        self.onlineGraph.step = framenum

        if valid:
            for pedID, itr in enumerate(current_batch):
                node_id = pedID
                node_pos_list = current_batch[pedID][int(framenum*self.ratio):int(framenum*self.ratio) + self.diff]  # * self.diff
                node = Node(node_id, node_pos_list, vislets)
                # if framenum > 0 and framenum % 8:
                #     node.setTargets(seq=future_traj[pedID])
                self.onlineGraph.setNodes(itr, node)
        else:
            self.pos_list_len = len(current_batch)
            for idx, itr in enumerate(current_batch):
            # for idx, itr in enumerate(current_batch['track_id']):
                try:
                    node_pos_list = current_batch[idx][int(framenum*self.ratio):int(framenum*self.ratio) + self.diff]  # * self.diff
                    node_id = pedID = idx #current_batch['track_id']
                    node = Node(node_id, node_pos_list, vislets[idx][int(framenum*self.ratio):int(framenum*self.ratio) + self.diff])
                    if not valid:
                        try:
                            if future_traj[pedID].shape[0] < 1:
                                node.setTargets(seq=future_traj[pedID][0:self.pred_diff])
                            elif future_traj[pedID][framenum:framenum + self.pred_diff].shape[0] < self.pred_diff:
                                node.setTargets(future_traj[pedID])
                            else:
                                node.setTargets(seq=future_traj[pedID][framenum:framenum + self.pred_diff])
                            self.onlineGraph.setNodes(itr, node)
                        except KeyError:
                            continue
                        except IndexError:
                            pass

                except KeyError:
                    key = list(current_batch.keys())
                    node_pos_list = current_batch[key[0]]

        self.onlineGraph.dist_mat = torch.zeros(len(self.onlineGraph.nodes), len(self.onlineGraph.nodes))
        # check online graph nodes are correct
        return self.onlineGraph

    def linkGraph(self, curr_graph, new_edges, frame):
        n1 = curr_graph.getNodes()
        for item in n1:
            edge_id = (item, item)
            dist = torch.norm(
                torch.from_numpy(np.subtract(n1[frame][item.id].pos[frame],
                                             n1[frame - 1][item.id].pos[frame - 1])), p=2)

            edge_weight = dist #{frame:dist}
            e = Edge(edge_id, edge_weight)
            curr_graph.setEdges(u=item.id, framenum=frame)
            curr_graph.setEdges(u= item.id , v=item.id, obj=e, mode='s')

    # Graph Networks (GN) formal functions
    def update(self):
        pass

    def aggregate(self):
        pass

class Graph(nx.Graph):
    def __init__(self, diff, pred_diff):
        super(Graph, self).__init__()
        self.adj_mat = []
        self.dist_mat = []

        # self.nodes = [{}]
        # self.edges = [[]]  # dict
        self.Stateful = True
        self.step = 0
        self.diff = diff
        self.pred_diff = pred_diff

        # by default the graph is stateful and each graph segment is connected to the previous temporal segment
        # unless nodes in a graph no longer exist in the scene, then we need to disconnect and destroy variables

    def getNodes(self):
        return self.nodes

    def getEdges(self):
        return self.edges

    def setNodes(self, framenum, node, pos_list_len=8):
        if node.id in self.nodes.keys():
            try:
                self.nodes[node.id]['node_pos_list'] = node.pos
            except IndexError:
                pass
        else:
            r_target = tf.repeat(input=node.targets[0][-1], repeats=(abs(len(node.targets[0]) - self.pred_diff)), axis=0)
            self.add_node(node.id,
                          seq=node.seq,
                          node_pos_list=tf.concat((node.pos, tf.zeros(shape=(abs(node.pos.shape[0] - self.diff), 2))), axis=0), #len(node.pos)
                          state=node.state,
                          cell=node.cell,
                          targets=tf.concat((node.targets[0], tf.reshape(r_target, (int(len(r_target)/2), 2))), axis=0),
                          vislets=tf.concat((node.vislet, tf.zeros(shape=(abs(node.vislet.shape[0] - self.diff), 2))), axis=0),
                          vel=node.vel)

    def setEdges(self, framenum , obj ,u,v=None, mode='t'):
        if mode == 't':
            nx.add_cycle(self.graph, self.nodes[u])
        else:
            if len(self.edges) <= framenum - 1:
                self.add_edge(u_for_edge=u, v_for_edge=v, key=str((u, v)), attr=obj.edge_weight)

        # print("appended new empty array")
        #     self.edges.append([])
        #     self.edges[framenum - 1].append(edge)
        # else:
        #     new_edge = Edge(edge.id , )

    def get_node_attr(self, param):
        return nx.get_node_attributes(G=self, name=param)

    def delGraph(g, framenum):
        g.clear()
        # del self.nodes
        # del self.edges
        # self.nodes = []
        # self.edges = []
        # for i in range(framenum):
        #     self.nodes.append({})
        #     self.edges.append([])

class Node():
    def __init__(self, node_id, node_pos_list, vislet=None):
        self.id = node_id
        # if len(self.pos):
        self.pos = node_pos_list
        self.state = torch.zeros(batch_size, 256)  # 256 self.human_ebedding_size
        self.cell = torch.zeros(batch_size, 256)
        self.seq = []
        self.targets = []
        if not vislet is None:
            self.vislet = vislet
        else:
            self.vislet = []
        self.vel = 0

    def setState(self, state, cell):
        self.state = state
        self.cell = cell

    def getState(self):
        return self.state, self.cell

    def setPrediction(self, seq):
        self.seq = seq

    def getPrediction(self):
        return self.seq

    def setTargets(self,seq):
        self.targets.append(seq)

    def getTargets(self):
        return self.targets

    def get_node_attr_dict(self):
        return {
                'seq': self.seq,
                'node_pos_list': self.pos,
                'state': self.state,
                'cell': self.cell,
                'targets': self.targets,
                'vislet': self.vislet,
                'vel': self.vel
                }


class Edge():
    def __init__(self, edge_id, edge_pos_list):
        self.id = edge_id
        self.dist = edge_pos_list



