---
layout: post
comments: false
title: "A Comprehensive Survey on Graph Neural Networks"
categories: misc
---

> Abstract: This work provides a comprehensive overview on graph nerual networks

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Definitions

We define a **graph** as $G = (V,E,A)$ where $A$ is the set of nodes, $E$ is the set of edges, and $A$ is the adjacency matrix. 

- **spatial-Temporal Graph** is an attributed graph where the feature matrix $X$ evolves. It is defined as $G=(V,E,A,X)$ with $X\in \mathbb{R}^{T\times N\times D}$ where $T$ is the length of time steps.

- **Graph Auto-encoder** employ GCN as the encoder and reconstruct the structure information via a link prediction decoder.
![Graph convolution network vs. graph attention network]({{'/assets/images/gcn_vs_gan.png'| relative_url}})

- **Graph Generative Network** one application is chemical compound synthesis. In a chemical graph, atoms are treated as nodes and bonds are treated as edges. The task is to discover new synthesize molecules which possess certain chemical and physical properties.

- **Graph spatial-temporal network** applications are traffic forecasting and human activity prediction. Many approaches apply Graph Convolutional Networks (GCNs) to capture the dependency together with some RNN or CNN to model the temporal dependency. 

## Training frameworks

- Semi-supervised learning for node-level classification
  - Given a single network with partial nodes being labeled and others remaining unlabeled, GCNs can learn a model that identify the class labels for the unlabeled nodes.

- Supervised learning for graph-level classification 
    - Given graph dataset, graph-level classification aims to predict the class label for an entire graph. The algorithm combines both GCN layers and the pooling procedure.
    ![Graph convolution network and pooling]({{'/assets/images/gcn_pooling.png'|relative_url}})

- Unsupervised learning for graph embedding
  - Adopt an autoencoder framework where the encoder employs graph convolution layers to embed the graph into the latent representation upon which a decoder is used to reconstruct the graph structure.
  - Utilize the negative sampling approach which samples a portion of node pairs as negative pairs while existing node pairs with links in the graphs being positive pairs.
  - Use logistic regression as a loss function.

## Graph Convolution networks
  
### Spectral based GCN
Graph are assumed to be undirected. A robust mathematical representation of an undirected graph is the normalized graph Laplacian matrix 

$$L=I_n - D^{-\frac{1}{2}}AD^{-\frac{1}{2}},$$

where $D_{ii} = \sum_j (A_{ij})$ is a diagonal matrix of node degree. The normalized graph Laplacian matrix can be factored as 

$$L=U\Lambda U^\top,$$

where eigen vectors $U=[u_0,u_1,...,u_{n-1}] \in \mathbb{R}^{N\times N}$, and eigen values $\Lambda_{ii} = \lambda_i$. 

The graph signal $x\in\mathbb{R}^N$ is a feature vector of the nodes of a graph, where $x_i$ is the value of the $i$-th node. The graph Fourier transform a signal $x$ is defined as 

$$F(x)=U^\top x,$$ 

and the inverse graph Fourier transform is defined as 

$$F^{-1}(F(x))=U F(x)$$ 

Therefore, the GCN of the input signal $x$ with a filter $g\in \mathbb{R}^N$ is defined as

$$
x *_G g = F^{-1}(F(x) \odot F(g)) = U(U^\top x \odot U^\top g)
$$

If we denote a filter as $g_\theta = \text{diag}(U^\top g)$, then the graph convolution is simplified as 

$$x *_G g = Ug_\theta U^\top x$$

Spectral-based GCNs all follow this definition. The key difference lies in the choice of the filter $g_\theta$. We drawbacks of Spectral-based GCNs:

- Any permutation to a graph results in a change of eigenbasis
- The learned filters are domain-dependent, meaning they cannot be applied to a graph with a different structure 
- Eigen-decomposition is computationally inefficient
- The hole graph need to be loaded into the memory to perform graph convolution, which is not efficient in handling big graphs


### Spatial based GCN
Spatial-based methods define graph convolutions via aggregating feature information from neighbors. According to different ways of stacking graph convolution layers, spatial-based methods are split into two categories

![Recurrent-based method vs. composition-based method]({{'/assets/images/gcn_recurrent_composition.png'|relative_url}})

- **Recurrent-based methods** applying the same graph convolution layer to update hidden representations, which obtains nodes' steady states
  - **Graph Neural Networks (GNNs)**
    To handle heterogeneous graphs, the spatial graph convolution of GNNs is defined as 

    $$
    h_v^t = f(I_v, I_{co}[v], h_{ne}^{t-1}[v], I_{ne}[v]),
    $$

    where $I_v$ denotes the label attributes of node $v$, $I_{co}[v]$ denotes the label attributes of corresponding edges of node $v$, $h_{ne}^{t}[v]$ denotes the hidden representations of node $v$'s neighbors at time $t$, and $I_{ne}[v]$ denotes the label attribute of the node $v$'s neighbors, $f(\cdot)$ is a neural network.
   
  - **Gated Graph Neural Networks (GGNNs)**
    It employs GRU as the recurrent function. The GGNN is defined as 

    $$
    h_v^t = \text{GRU}(h_v^{t-1}, \sum_{u\in N(v)} Wh_u^t)
    $$

  - **Stochastic Steady-state Embedding (SSE)**
    To improve the learning efficiency, the SSE algorithm updates the node latent representations stochastically in an asynchoronous fashion. SSE recursively estimates node latent representations and updates the parameters with sampled batch data. 

    $$
    h_v^t = (1-\alpha)h_v^{t-1} + \alpha W_1 \sigma(W_2[x_v, \sum_{u\in N(v)}[h_u^{t-1},x_u]])
    $$

    ![Stochastic Steady-state Embedding]({{'/assets/images/SSE.png'|relative_url}})
- **Composition-based methods** apply a different graph convolution layer to update hidden representation, which incorporate higher orders of neighborhood information.

  Composition-based methods updates the node representation by stacking multiple graph convolution layers.

  - **Message Passing Neural Networks (MPNNs)**
    The MPNNs consists of two phases, the message passing phase and the readout phase. The message passing phase runs $T$-step spatial-based graph convolutions

    $$
    h_v^t = U_t (h_v^{t-1}, \sum_{u\in N(v)} M_t(h_v^{t-1}, h_u^{t-1}, e_{vu})),
    $$
    where $M(\cdot)$ is message function and $U_t(\cdot)$ is the updating function.

    The readout phase is actually a pooling operation which produces a representation of the entire graph based on hidden representations of each individual node. It is defined as 

    $$
    y = R(h_v^\top | v \in G)
    $$

    The output $y$ is used to perform graph-level prediction tasks. 

  - **GraphSage**
    GraphSage introduces notion of the aggregation function to define graph convolution. The aggregation function essentially assembles a node's neighborhood information. 

    $$
    h_v^t = \sigma (W^t \cdot \text{aggregate}_t(h_v^{t-1}, \{h_u^{t-1}, \forall u\in N(v)\}))
    $$

    Instead of updating states over all nodes, GraphSage proposes a batch-training algorithm, which improves scalability for large graphs. The training process contains three steps. 

    - First, it samples a node's local $k$-hop neighborhood with fixed size. 
    - Second, it derives the central node's final state by aggregating its neighbors feature information. 
    - Finally, it uses the central node's final state to make predictions and backpropagate errors.
    ![GraphSage]({{'/assets/images/graphsage.png'|relative_url}})

- Others
  - Diffusion Convolution Neural Networks (DCNN)
  - PATCHY-SAN
  - Large-scale Graph Convolution Networks (LGCN)
  - Mixture Model Network (MoNet)

### Graph pooling modules
- **SortPooling**
- **DIFFPOOL**
  DIFFPOOL is proposed which can generate hierarchical representations of graphs. Compared with other methods, DIFFPOOL does not simply cluster the nodes in one graph but provide a general solution to hierarchically pool nodes across a broad set of input graphs. This is done by learning a cluster assignment matrix $S^{(l)} \in \mathbb{R}^{n_l \times n_{l + 1}}$ at layer $l$. Two separate GNNs are used to generate assignment matrices $S^{(l)}$ and embedding matrices $Z^{(l)}$ as follows

  $$
  \begin{aligned}
  Z^{(l)} &= \text{GNN}_{l, \text{embed}} (A^{(l)}, X^{(l)}) \\
  S^{(l)} &= \text{Softmax}(\text{GNN}_{l, \text{pool}}(A^{(l)}, X^{(l)})) \\
  \end{aligned}
  $$

  As a result, each row of $S^{(l)}$ corresponds to one of the $n_l$ nodes (or clusters) at layer $l$, and each column of $S^{(l)}$ corresponds to one of the $n_l$ at the next layer. Given assignment matrices $S^{(l)}$ and embedding matrices $Z^{(l)}$ as follows, the pooling operation comes as follows:

  $$
  \begin{aligned}
  X^{(l+1)} &= {S^{(l)}}^\top  Z^{(l)} \in \mathbb{R}^{n_{l+1} \times d} \\
  A^{(l+1)} &= {S^{(l)}}^\top A^{(l)} S^{(l)} \in \mathbb{R}^{n_{l+1} \times n_{l+1}}
  \end{aligned}
  $$

### Spectral vs. Spatial
There are several drawbacks to spectral-based models. Here we illustrate this in the following from three aspects, efficiency, generality, and flexibility.

- In terms of efficiency, the computational cost of spectral-based models increases dramatically with the graph size because they either need to perform eigenvector computation or handle the whole graph at the same time, which makes them difficult to parallel or scale to large graph
- In terms of generality, spectral-based models assumed a fixed graph, making them generalize poorly to new or different graphs
- In terms of flexibility, spectral-based models are limited to work on undirected graphs. There is no clear definition of the Laplacian matrix on directed graphs so that the only way to apply spectral-based models to directed graphs is to transfer directed graphs to undirected graphs.

## Graph Attention Networks
- **Graph Attention Network (GAT)**

    $$
    h_i^t = \sigma \left(\sum_{j \in N(i)} \alpha(h_i^{t-1}, h_j^{t-1}) W^{t-1} h_j^{t-1}\right)
    $$

- **Gated Attention Network (GAAN)**

    $$
    h_i^t = \phi_o \left(x_i \oplus \|_{k=1}^K g_i^k \sum_{j \in N(i)} \alpha_k(h_i^{t-1}, h_j^{t-1})\phi_v(h_j^{t-1})\right)
    $$

- **Attention Walks**
    - Learns node embeddings through random walks. Attention Walks factorizes the co-occurrence matrix with differentiable attention weights 

Attention mechanisms help GNNs by 
- assigning attention weights to different neighbors when aggregating feature information
- ensembling multiple models according to attention weights
- using attention weights to guide random walks

## Graph Auto-encoders
Graph auto-encoders are one class of network embedding approaches which aim at representing network vertices into a low-dimensional vector space by using neural network architectures. A typical solution is to leverage multi-layer perceptions as the encoder to obtain node embeddings, where a decoder reconstructs a node's neighborhood statistics.

- **Graph Auto-encoder (GAE)**
    The encoder is defined as 

    $$
    Z = \text{GCN}(X,A),
    $$

    and the decoder is defined as 

    $$
    A = \sigma(Z^\top Z)
    $$

    The GAE can be trained in a variational manner

    $$
    L = \mathbb{E}_{q(Z|X,A)}[\log_p(A|Z)] - \text{KL}[q(Z|X,A) \| p(Z)]
    $$

## Graph Spatial-Temporal Networks
Graph spatial-temporal networks capture spatial and temporal dependencies of a spatial-temporal graph simultaneously. Here we introduce some GCN based graph spatial-temporal networks.

- Diffusion Convolutional Recurrent Neural Network (DCRNN)
  - introduces diffusion convolution as graph convolution for capturing spatial dependency and use sequence-to-sequence architecture with gated recurrent units (GRU) to capture temporal dependency.
- CNN-GCN
  - interleaves 1D-CNN with GCN to learn spatial-temporal graph data. 
- Spatial Temporal GCN (ST-GCN)
  - adopts a different approach by extending the temporal flow as graph edges so that spatial and temporal information can be extracted using a unified GCN model at the same time.
  - ST-GCN defines a function to assign label to each edge of the graph according to the distance of the two related nodes.
  - The adjacency matrix can be represented as a summation of $K$ adjacent matrices where $K$ is the number of labels.
  - Then, ST-GCN applies GCN with a different weight matrix to each $K$ adjacent matrix and sums them.

  $$
  f_{out} = \sum_j D_j^{-\frac{1}{2}} A_j D_j^{-\frac{1}{2}} f_{in} W_j
  $$

## Further Directions
- Go Deep
  It has been shown that with the increase in the number of layers, the gnn performance drops dramantically. This is due to the effect of gcn pushes representations of adjacent nodes closer to each other so that with an infinite times of convolutions, all nodes' representations will converge to a single point.

- Receptive field
  The receptive field of a node refers to a set of nodes including the central node and its neighbors. How to select a representative receptive field of a node remains to be explored

- Scalability
  Most gnn do not scale well for large graphs. By stacking multiple layers of a graph convolution, a node's final state involves a large number of its neighbors' hidden states, leading to the high complexity of backpropagation.

  - fast sampling: 
    - Fastgcn: fast learning with graph convolutional networks via importance sampling
    - Stochastic training of graph convolutional networks with variance reduction
  - subgraph training:
    - Inductive representation learning on large graphs
    - Large-sclae learnable graph convolutional network
  - Dynamics and Heterogeneity
      - GNN assume graph structures are fixed, and nodes and edges from a graph are assumed to come from a single source.