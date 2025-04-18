import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch_geometric
import sys

sys.path.insert(0, "./graph_construct")
from .constants import NODE_TYPES, EDGE_TYPES, CONV_LAYERS, NORM_LAYERS, RESIDUAL_LAYERS, NODE_TYPE_TO_LAYER
from .utils import (
    make_node_feat,
    make_edge_attr,
    conv_to_graph,
    linear_to_graph,
    norm_to_graph,
    ffn_to_graph,
    basic_block_to_graph,
    self_attention_to_graph,
    equiv_set_linear_to_graph,
    triplanar_to_graph,
)
from .layers import (
    Flatten,
    PositionwiseFeedForward,
    BasicBlock,
    SelfAttention,
    EquivSetLinear,
    TriplanarGrid,
)
import matplotlib.pyplot as plt

# model <--> arch <--> graph



def seq_to_feats(seq: nn.Sequential):
    """
    Convert a sequential model to node features and edge attributes.

    Args:
        seq (torch.nn.Sequential): The sequential model to convert.

    Returns:
        torch.Tensor: The node feature matrix - num_nodes x 3 node features
        torch.Tensor: The edge attribute matrix - num_edges x 6 edge features

    """
    return arch_to_graph(sequential_to_arch(seq))

def sequential_to_arch(model):
    """
    Convert a sequential model to an architecture, which is a list of lists where each list contains the
        layer type and the weights and biases of the layer.
    Args:
        model (torch.nn.Sequential): The sequential model to convert.
    Returns:
        List[List[torch.nn.Module, torch.Tensor, torch.Tensor]]: The architecture of the model.
            - The first element of each list is the layer type.
            - The second element of each list is the weight tensor.
            - The third element of each list is the bias tensor.
    """
    # input can be a nn.Sequential
    # or ordered list of modules
    arch = []
    weight_bias_modules = CONV_LAYERS + [nn.Linear] + NORM_LAYERS
    for i, module in enumerate(model):
        layer = [type(module)]
        if type(module) in weight_bias_modules:
            weight = module.weight
            bias = module.bias
            
            layer.append(weight)
            layer.append(bias)
        else:
            if len(list(module.parameters())) != 0:
                raise ValueError(
                    f"{type(module)} has parameters but is not yet supported"
                )
            continue
        layer.append(i)
        arch.append(layer)
    
    return arch


def arch_to_graph(arch, self_loops=False):
    """
    Convert an architecture to a graph, which is represented by node features, edge indices, and edge attributes.
    This version ensures the weights and biases remain in the computation graph.

    Args:
        arch (List[List[torch.nn.Module, torch.Tensor, torch.Tensor]]): The architecture of the model.
            - The first element of each list is the layer type.
            - The second element of each list is the weight tensor.
            - The third element of each list is the bias tensor.
        self_loops (bool, optional): Whether to include self loops. Defaults to False.

    Returns:
        torch.Tensor: The node feature matrix - num_nodes x 3 node features
        torch.Tensor: The edge indices - 2 x num_edges (source, target)
        torch.Tensor: The edge attribute matrix - num_edges x 6 edge features
    """
    curr_idx = 0  # used to keep track of current node index relative to the entire graph
    node_features = []  # stores a list of tensors, each representing the features of a node
    edge_index = []  # stores a list of tensors, each stores 2xnum_edges (source, target)
    edge_attr = []  # stores a list of tensors, each stores num_edges x 6 edge features
    layer_num = 0  # keep track of current layer number

    # initialize input nodes
    layer = arch[0]
    layer_type = layer[0]
    if layer_type in CONV_LAYERS:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer_type in (nn.Linear, PositionwiseFeedForward):
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer_type == BasicBlock:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer_type == EquivSetLinear:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer_type == TriplanarGrid:
        triplanar_resolution = layer[1].shape[2]
        in_neuron_idx = torch.arange(3 * triplanar_resolution**2)
    else:
        raise ValueError("Invalid first layer")

    for i, layer in enumerate(arch):
        is_output = i == len(arch) - 1
        layer_type = layer[0]
        
        
        if layer_type in CONV_LAYERS:
            weight_mat, bias = layer[1], layer[2]
            ret = conv_to_graph(
                weight_mat,
                bias,
                layer_num,
                in_neuron_idx,
                is_output,
                curr_idx,
                self_loops,
            )
            layer_num += 1
        elif layer_type == nn.Linear:
            weight_mat, bias = layer[1], layer[2]
            ret = linear_to_graph(
                weight_mat,
                bias,
                layer_num,
                in_neuron_idx,
                is_output,
                curr_idx,
                self_loops,
            )
            layer_num += 1
        elif layer_type in NORM_LAYERS:
            if layer_type in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
                norm_type = "bn"
            elif layer_type == nn.LayerNorm:
                norm_type = "ln"
            elif layer_type == nn.GroupNorm:
                norm_type = "gn"
            elif layer_type in (
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
            ):
                norm_type = "in"
            else:
                raise ValueError("Invalid norm type")
            gamma = layer[1]
            beta = layer[2]
            ret = norm_to_graph(
                gamma,
                beta,
                layer_num,
                in_neuron_idx,
                is_output,
                curr_idx,
                self_loops,
                norm_type=norm_type,
            )
        elif layer_type == BasicBlock:
            ret = basic_block_to_graph(
                layer[1:], layer_num, in_neuron_idx, is_output, curr_idx, self_loops
            )
            layer_num += 2
        elif layer_type == PositionwiseFeedForward:
            ret = ffn_to_graph(
                layer[1],
                layer[2],
                layer[3],
                layer[4],
                layer_num,
                in_neuron_idx,
                is_output,
                curr_idx,
                self_loops,
            )
            layer_num += 2
        elif layer_type == SelfAttention:
            ret = self_attention_to_graph(
                layer[1],
                layer[2],
                layer[3],
                layer[4],
                layer_num,
                in_neuron_idx,
                is_output=is_output,
                curr_idx=curr_idx,
                self_loops=self_loops,
            )
            layer_num += 2
        elif layer_type == EquivSetLinear:
            ret = equiv_set_linear_to_graph(
                layer[1],
                layer[2],
                layer[3],
                layer_num,
                in_neuron_idx,
                is_output=is_output,
                curr_idx=curr_idx,
                self_loops=self_loops,
            )
            layer_num += 1
        elif layer_type == TriplanarGrid:
            ret = triplanar_to_graph(
                layer[1], layer_num, is_output=is_output, curr_idx=curr_idx
            )
            layer_num += 1
        else:
            raise ValueError("Invalid layer type")
        in_neuron_idx = ret["out_neuron_idx"]

        edge_index.append(ret["edge_index"])  # 2 x num_edges
        edge_attr.append(ret["edge_attr"])  # num_edges x 6
        if ret["node_feats"] is not None:
            feat = ret["node_feats"]
            node_features.append(feat)
            curr_idx += feat.shape[0]

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    edge_attr = torch.cat(edge_attr, dim=0)
    
    return node_features, edge_index, edge_attr

def feats_to_arch(node_features):
    arch = {}
    for i in range(node_features.shape[0]):
        node_feats = node_features[i]
        layer_num, _, node_type =  node_feats
        layer_num = layer_num.item()
        node_type = node_type.item()
        if layer_num in arch:
            continue

        arch[layer_num] = NODE_TYPE_TO_LAYER[node_type]
    arch = [arch[i] for i in range(len(arch))]

    return arch

def graph_to_arch(arch, weights):
    arch_new = []
    curr_idx = 0
    for l, layer in enumerate(arch):
        lst = [layer[0]]
        for tensor in layer[1:-1]: # ignore first elem (layer type) and last elem (layer number)
            if tensor is not None:
                weight_size = math.prod(tensor.shape)
                reshaped = weights[curr_idx : curr_idx + weight_size].reshape(tensor.shape) 
                lst.append(reshaped)
                curr_idx += weight_size
        lst.append(layer[-1]) # append layer number
        arch_new.append(lst)
    return arch_new


def arch_to_named_params(arch):
    '''
    arch: the architecture of the model, as a list of lists

    returns a generator of tuples of (name, param)
    '''

    
    for i, layer in enumerate(arch):
        layer_num = layer[-1]
        yield f'{layer_num}.weight', layer[1]
        yield f'{layer_num}.bias', layer[2]



def visualize_graph(x, edge_index, edge_attr):
    import networkx as nx
    from torch_geometric.utils import to_networkx

    data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    G = to_networkx(data)
    for node in range(x.shape[0]):
        layer_num = x[node, 0].item()
        # use layer num as subset key
        G.nodes[node]["subset"] = layer_num
    pos = nx.multipartite_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()


def tests():
    import networkx as nx
    from torch_geometric.utils import to_networkx
    from net_makers import make_transformer, make_resnet

    def test1(model):
        """
        Test the graph construction from a sequential model.

        Args:
            model (torch.nn.Sequential): The sequential model to test.

        Raises:
            AssertionError: If any of the assertions fail.

        """
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)

        # number of edges consistent
        assert edge_index.shape[1] == edge_attr.shape[0]

        # nodes in edge index exist
        assert x.shape[0] == edge_index.max() + 1

        # each node is in some edge
        assert (torch.arange(x.shape[0]) == edge_index.unique()).all()

        num_params = sum([p.numel() for p in model.parameters()])
        # at least one edge for each param (not exact because residuals)
        assert edge_index.shape[1] >= num_params

    def test2(model):
        """
        Test the graph construction from a sequential model, and then reconstruction of the model.
        """
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        new_arch = graph_to_arch(arch, edge_attr[:, 0])
        new_model = arch_to_sequential(arch, deepcopy(model))

        # make sure original objects are reconstructed correctly
        for i in range(len(arch)):
            for j in range(1, len(arch[i])):
                eq = arch[i][j] == new_arch[i][j]
                if type(eq) == torch.Tensor:
                    eq = eq.all()
                assert eq

        # check state dicts are the same
        sd1, sd2 = model.state_dict(), new_model.state_dict()
        for k, v in sd1.items():
            assert (v == sd2[k]).all()

    def test3(model, plot=False):
        """
        Test the graph construction from a sequential model, and then check the graph is a DAG and weakly connected.
        """
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        data = torch_geometric.data.Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr
        )
        G = to_networkx(data)

        # G should be a DAG
        assert nx.is_directed_acyclic_graph(G)

        # G should be weakly connected
        assert nx.is_weakly_connected(G)

        if plot:
            visualize_graph(x, edge_index, edge_attr)

    def test4():
        # hard coded some small neural networks

        model = nn.Sequential(BasicBlock(1, 1))
        model(torch.randn(16, 1, 5, 4))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 7
        assert num_params == 22
        assert edge_index.shape[1] == edge_attr.shape[0] == 23
        assert (
            edge_index
            == torch.tensor(
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        2,
                        3,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        5,
                        6,
                        0,
                    ],
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                    ],
                ]
            )
        ).all()

        model = nn.Sequential(BasicBlock(1, 2))
        model(torch.randn(16, 1, 5, 4))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 13
        assert num_params == 68
        assert edge_index.shape[1] == edge_attr.shape[0] == 70
        # lmao i actually wrote this all out by hand
        expected = [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                3,
                3,
                4,
                4,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                7,
                7,
                8,
                8,
                0,
                0,
                11,
                11,
                12,
                12,
                9,
                10,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                2,
                1,
                2,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                5,
                6,
                5,
                6,
                9,
                10,
                9,
                10,
                9,
                10,
                5,
                6,
            ],
        ]
        visualize_graph(x, edge_index, edge_attr)
        edge_index_np = edge_index.cpu().numpy()
        for i in range(len(expected[0])):
            if edge_index[0, i] != expected[0][i] or edge_index[1, i] != expected[1][i]:
                print(
                    i,
                    edge_index_np[0, i],
                    expected[0][i],
                    edge_index_np[1, i],
                    expected[1][i],
                )
            assert (
                edge_index_np[0, i] == expected[0][i]
                and edge_index_np[1, i] == expected[1][i]
            )
        assert (
            edge_index
            == torch.tensor(
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        3,
                        3,
                        4,
                        4,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        7,
                        7,
                        8,
                        8,
                        0,
                        0,
                        11,
                        11,
                        12,
                        12,
                        9,
                        10,
                    ],
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        1,
                        2,
                        1,
                        2,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        5,
                        6,
                        5,
                        6,
                        9,
                        10,
                        9,
                        10,
                        9,
                        10,
                        5,
                        6,
                    ],
                ]
            )
        ).all()

        model = nn.Sequential(
            nn.Linear(2, 3, bias=False), nn.ReLU(), nn.LayerNorm(3), nn.Linear(3, 1)
        )
        model(torch.randn(16, 2))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 9
        assert num_params == 16
        assert edge_index.shape[1] == edge_attr.shape[0] == 16
        assert (
            edge_index
            == torch.tensor(
                [
                    [0, 0, 0, 1, 1, 1, 5, 5, 5, 6, 6, 6, 2, 3, 4, 8],
                    [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 7, 7, 7, 7],
                ]
            )
        ).all()

        model = nn.Sequential(
            nn.Conv1d(1, 3, 2, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(3),
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(3, 2),
        )
        model(torch.randn(16, 1, 5))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 9
        assert num_params == 20
        assert edge_index.shape[1] == edge_attr.shape[0] == 20
        assert (
            edge_index
            == torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 4, 4, 4, 5, 5, 5, 1, 1, 2, 2, 3, 3, 8, 8],
                    [1, 1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3, 6, 7, 6, 7, 6, 7, 6, 7],
                ]
            )
        ).all()

        # small attention layer
        model = nn.Sequential(
            nn.Linear(1, 1, bias=False), nn.LayerNorm(1), SelfAttention(1, 1)
        )
        model(torch.randn(16, 5, 1))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 10
        assert num_params == 11
        assert edge_index.shape[1] == edge_attr.shape[0] == 12
        assert (
            edge_index
            == torch.tensor(
                [
                    [0, 2, 3, 1, 5, 1, 6, 1, 7, 4, 9, 1],
                    [1, 1, 1, 4, 4, 4, 4, 4, 4, 8, 8, 8],
                ]
            )
        ).all()

        model = make_transformer(1, 1, 1, 1, num_layers=1, vit=False)
        model(torch.randn(16, 8, 1))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 22
        assert num_params == 29
        assert edge_index.shape[1] == edge_attr.shape[0] == 31
        expected_edge_index = torch.tensor(
            [
                [
                    0,
                    2,
                    3,
                    4,
                    1,
                    6,
                    1,
                    7,
                    1,
                    8,
                    5,
                    10,
                    1,
                    11,
                    12,
                    9,
                    9,
                    9,
                    9,
                    17,
                    17,
                    17,
                    17,
                    13,
                    14,
                    15,
                    16,
                    19,
                    9,
                    18,
                    21,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    9,
                    9,
                    9,
                    9,
                    9,
                    13,
                    14,
                    15,
                    16,
                    13,
                    14,
                    15,
                    16,
                    18,
                    18,
                    18,
                    18,
                    18,
                    18,
                    20,
                    20,
                ],
            ]
        )
        assert (edge_index == expected_edge_index).all()

        model = nn.Sequential(
            EquivSetLinear(1, 2), nn.GroupNorm(1, 2), nn.ReLU(), EquivSetLinear(2, 1)
        )
        assert model(torch.randn(16, 1, 4)).shape == (16, 1, 4)
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 8
        assert num_params == 15
        assert edge_index.shape[1] == edge_attr.shape[0] == 15
        assert (
            edge_index
            == torch.tensor(
                [
                    [0, 0, 3, 3, 0, 0, 4, 4, 5, 5, 1, 2, 7, 1, 2],
                    [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 6, 6, 6, 6, 6],
                ]
            )
        ).all()

        model = nn.Sequential(TriplanarGrid(2, 1), nn.ReLU(), nn.Linear(4, 1))
        assert model(torch.randn(10, 3) * 0.1).shape == (10, 1)
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 18
        assert num_params == 17
        assert edge_index.shape[1] == edge_attr.shape[0] == 17
        assert (
            edge_index
            == torch.tensor(
                [
                    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 15, 17],
                    [
                        15,
                        15,
                        15,
                        15,
                        15,
                        15,
                        15,
                        15,
                        15,
                        15,
                        15,
                        15,
                        16,
                        16,
                        16,
                        16,
                        16,
                    ],
                ]
            )
        ).all()

        model = nn.Sequential(TriplanarGrid(2, 2), nn.ReLU(), nn.Linear(5, 1))
        assert model(torch.randn(10, 3) * 0.1).shape == (10, 1)
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 19
        assert num_params == 30
        assert edge_index.shape[1] == edge_attr.shape[0] == 30
        assert (
            edge_index
            == torch.tensor(
                [
                    [
                        3,
                        3,
                        4,
                        4,
                        5,
                        5,
                        6,
                        6,
                        7,
                        7,
                        8,
                        8,
                        9,
                        9,
                        10,
                        10,
                        11,
                        11,
                        12,
                        12,
                        13,
                        13,
                        14,
                        14,
                        0,
                        1,
                        2,
                        15,
                        16,
                        18,
                    ],
                    [
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        15,
                        16,
                        17,
                        17,
                        17,
                        17,
                        17,
                        17,
                    ],
                ]
            )
        ).all()

    model1 = nn.Sequential(
        nn.Linear(2, 3), nn.SiLU(), nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 2)
    )
    model2 = nn.Sequential(
        nn.Linear(2, 3),
        nn.LayerNorm(3),
        nn.SiLU(),
        nn.Linear(3, 3),
        nn.ReLU(),
        nn.Linear(3, 2),
    )
    model3 = nn.Sequential(
        nn.Conv2d(3, 4, 3),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 5, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.LayerNorm(5),
        nn.Linear(5, 1),
    )
    model4 = nn.Sequential(PositionwiseFeedForward(10, 20), nn.ReLU(), nn.Linear(10, 4))
    model5 = nn.Sequential(
        nn.Conv2d(2, 3, 3, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, 3, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 1, bias=False),
    )

    model6 = nn.Sequential(
        nn.Conv2d(3, 2, 3),
        nn.ReLU(),
        BasicBlock(2, 4),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )
    model7 = nn.Sequential(
        nn.Conv2d(3, 2, 3),
        nn.ReLU(),
        BasicBlock(2, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )
    model8 = nn.Sequential(
        nn.Linear(1, 4, bias=False),
        nn.LayerNorm(4),
        SelfAttention(4, 1),
        PositionwiseFeedForward(4, 16),
    )
    model9 = make_transformer(3, 16, 4, 2, num_layers=3, vit=True, patch_size=4)
    model10 = make_transformer(3, 16, 4, 2, num_layers=3, vit=False)
    model11 = make_resnet(conv_layers=2, hidden_dim=8, in_dim=3, num_classes=4)
    model12 = nn.Sequential(EquivSetLinear(2, 3), nn.ReLU(), EquivSetLinear(3, 1))
    model13 = nn.Sequential(
        TriplanarGrid(4, 2), nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1)
    )
    models = [
        model1,
        model2,
        model3,
        model4,
        model5,
        model6,
        model7,
        model8,
        model9,
        model10,
        model11,
        model12,
        model13,
    ]

    for i, model in enumerate(models):
        print("Model:", i + 1)
        test1(model)
        test2(model)
        test3(model, plot=False)

    test4()

    print("Tests pass!")


if __name__ == "__main__":
    tests()
