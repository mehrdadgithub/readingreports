---
layout: post
comments: false
title: "Self-Attention Graph Pooling"
categories: misc
---

> Abstract: In this paper, we propose a graph pooling method based on self-attention. Self-attention using graph convolution allows the pooling algorithm to consider both node features and graph topology.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Recently, DIFFPOOL [1] is proposed which can learn hierarchical representation of graphs. However, DIFFPOOL has a quadratic storage complexity and the number of its parameters is dependent on the number of nodes. Graph U-net [2] have addressed the complexity issue, but their method does not take graph topology into account.

To overcome the limitations, SAGPool, a self-attention graph pooling method for GNNs in the context of hierarchical graph pooling, is proposed. The self-attention mechanism is exploited to distinguish between the nodes that should be dropped and the node that should be retained. Due to the self-attention mechanism which uses graph convolution to calculate attention scores, node features and graph topology are considered.

![sagpool framework]({{'/assets/images/sagpool.png'|relative_url}})

## Method

### Self-attention Graph Pooling

The self-attention scores are obtained using graph convolution. The self-attention score $Z\in\mathbb{R}^{N\times 1}$ is calculated as
    
$$
Z = \sigma(\text{GNN}(A,X) )
$$

where $\sigma$ is the activation function. 

The top $\lceil kN\rceil$ nodes are selected based on the value of $Z$.

$$
\text{idx} = \text{top-rank}(Z, \lceil kN\rceil)
$$

where top-rank function returns the indices of the top $\lceil kN\rceil$ values.

The new node embeddings are defined as $X' = X_{idx}$ and the new adjacent matrix is defined as $A' = A_{idx,idx}$.

### Global and hierarchical pooling

![Global and hierarchical pooling]({{'/assets/images/global_hierarchical.png'|relative_url}})



## Reference
[1] Ying, Zhitao, et al. "Hierarchical graph representation learning with differentiable pooling." Advances in Neural Information Processing Systems. 2018.

[2] Gao, Hongyang, and Shuiwang Ji. "Graph U-Nets." arXiv preprint arXiv:1905.05178 (2019).	
