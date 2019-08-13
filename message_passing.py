import tensorflow as tf


def message_passing(nodes, edges, edge_features, message_fn, edge_keep_prob=1.0):
    """
    Pass messages between nodes and sum the incoming messages at each node.
    Implements equation 1 and 2 in the paper, i.e. m_{.j}^t &= \sum_{i \in N(j)} f(h_i^{t-1}, h_j^{t-1})

    :param nodes: (n_nodes, n_features) tensor of node hidden states.
    :param edges: (n_edges, 2) tensor of indices (i, j) indicating an edge from nodes[i] to nodes[j].
    :param edge_features: features for each edge. Set to zero if the edges don't have features.
    :param message_fn: message function, will be called with input of shape (n_edges, 2*n_features + edge_features). The output shape is (n_edges, n_outputs), where you decide the size of n_outputs
    :param edge_keep_prob: The probability by which edges are kept. Basically dropout for edges. Not used in the paper.
    :return: (n_nodes, n_output) Sum of messages arriving at each node.
    """
    n_nodes = tf.shape(nodes)[0] # number of nodes, type: scalar Tensor
    n_features = nodes.get_shape()[1].value # number of features of nodes, type: python int
    n_edges = tf.shape(edges)[0] # number of edges, type: scalar Tensor

    message_inputs = tf.gather(nodes, edges)  # shape: n_edges, 2, n_features
    # reshape operation will concatenate the two corresponding nodes' hidden states
    reshaped = tf.concat([tf.reshape(message_inputs, (-1, 2 * n_features)), edge_features], 1)
    messages = message_fn(reshaped)  # shape: n_edges, n_output
    messages = tf.nn.dropout(messages, edge_keep_prob, noise_shape=(n_edges, 1)) # noise_shape=(n_edges, 1) means drop or keep the entire row

    n_output = messages.get_shape()[1].value

    idx_i, idx_j = tf.split(edges, 2, 1) #  edges is split along dimension 1 into 2_smaller tensors.
    # shape of idx_j: n_edges, 1

    out_shape = (n_nodes, n_output)
    # sum the incoming message at each node
    updates = tf.scatter_nd(idx_j, messages, out_shape)
    # the n_row of out_shape is n_nodes(< n_edges), so after scatter the tensor in 'messages' into out_shape according to idx_j
    # element of out_shape will be the sum of those tensor which are assigned to the same position, see the following example
    '''
    indices = tf.constant([[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3]])
    updates = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12])
    shape = tf.constant([4])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
    print(sess.run(scatter))
    # output is [15 18 21 24], 15=1+5+9, 18=2+6+10 and so on
    '''

    return updates
