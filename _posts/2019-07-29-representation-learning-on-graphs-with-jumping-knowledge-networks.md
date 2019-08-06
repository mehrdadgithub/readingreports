---
layout: post
comments: false
title: "Representation Learning on Graphs with Jumping Knowledge Networks"
categories: misc
---

> Abstract: This paper analyze the limitations of neighborhood aggregation based GNNs - the range of "neighboring" nodes that a node's representation draws from strongly depends on the graph structure - and propose a strategy to overcome the limitations. To adapt to local neighborhood properties and task, this work explore an architecture, jumping knowledge (JK) networks, that flexibly leverages different neighborhood ranges to enable better structure-aware representation.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Representation learning of nodes in graphs aims to extract high-level features from a node as well as its neighborhood, and has proved extremely useful for many applications. 

Many of these approaches broadly follow a neighborhood aggregation. Theoretically, an aggregation process of $k$ iterations makes use of the subtree structures of hight $k$ rooted at every node. Such schemes have been shown to generalize the WL test enabling to simultaneously learn the topology as well as the distribution of node features in the neighborhood.

Yet, such aggregation schemes sometimes lead to surprises. For example, the deeper the GNN, the worse the performance. Motivated by observations like this, we address two questions. 

- First, we study properties and resulting limitations of neighborhood aggregation schemes. 
- Second, based on this analysis, we propose an architecture that enables adaptive structure-aware representations.

### Model analyze
We define **influence distribution** as the effective range of nodes that any given node's representation draws from. This effective range implicitly encodes prior assumptions on what are the "nearest neighbors" that a node should draw information from. In particular, we will see that this influence is heavily affected by the graph structure.

### Changing locality
Many real-world graphs possess locally strongly varying structure.  In biological and citation networks, the majority of the nodes have few connections, whereas some nodes (hubs) are connected to many other nodes. Social and web networks usually consist of an expander-like core part and an almost-tree (bounded tree-width) part, which represent well-connected entities and the small communities respectively

Besides node features, this subgraph structure has great impact on the result of neighborhood aggregation. The speed of expansion (growth of the influence radius) is characterized by the random walk's mixing time, which changes dramatically on subgraphs with different structures.

### JK networks
The above observations raise the question whether it is possible to adaptively adjust the influence radii for each node and task. To achieve this, we explore an architecture that learns to selectively exploit information from neighborhoods of differing locality. This architecture selectively combines different aggregations at the last layer.

## Preliminary

### Notations
We define $G=(V,E)$ be a simple graph with node features $X_v \in \mathbb{R}^{d_i}$. The hidden state learned by the $l$-th layer of the model is denoted by $h^{(l)}_v \in \mathbb{R}^{d_h}$.

**Standard neighborhood aggregation**
A typical neighborhood aggregation scheme can be written as (graph convolutional network)

$$
h_v^{(l)} = \sigma \left(W_l \cdot \text{aggregate}(\{h_u^{(l-1)}, u\in N(v)\})\right)
$$

**Neighborhood aggregation with skip connections**

$$
\begin{aligned}
h_{N(v)}^{(l)} &= \sigma \left(W_l \cdot \text{aggregate}(\{h_u^{(l-1)}, u\in N(v)\})\right)\\
h_v^{(l)} &= \text{combine}(h_v^{(l-1)}, h_{N(v)}^{(l)})
\end{aligned}
$$

The skip connections are input-unit specific but not output-unit specific. That is, if we skip a layer for $h_v^{(l)}$ do not aggregate or use a certain "combine" operation, all subsequent units using this representation will be using this skip implicitly. It is impossible that a certain higher-up representation $h_u^{(l+j)}$ uses this skip and another one does not. 

Therefore, skip connections cannot adaptively adjust the neighborhood sizes of the final-layer representations independently.

**Neighborhood aggregation with directional biases**

Some recent models using attention weight for "important" neighbors. This can be seen as neighborhood aggregation with directional biases because some directions of expansion more than the others.

## Influence Distribution and Random Walks

We measure the sensitivity of node $x$ to node $y$ by measuring how much a change in the input feature of $y$ affects the representation of $x$ in the last layer. For node $x$, we define the influence score and distribution as follows

> (Influence score and distribution) For a graph $G=(V,E)$, let $h_x^{(0)}$ be the input feature and $h_x^{(k)}$ be the learned hidden feature of node $x\in V$ at the $k$-th layer of the model. The influence score $I(x,y)$ as 
> 
> $$
> I(x,y) = \frac{\partial h_x^{(k)}}{\partial h_y^{(0)}}
> $$ 
> 
> and define the influence distribution $I_x(y)$ by normalizing the influence scores 
> 
> $$
> I_x(y) = e^\top [\frac{\partial h_x^{(k)}}{\partial h_y^{(0)}}] e / \left( \sum_{z\in V} e^\top [\frac{\partial h_x^{(k)}}{\partial h_z^{(0)}}] e \right)
> $$

Here we define the random work distribution. Consider a random walk on $G$ starting at a node $v_0$. If at the $t$-th the step we are at a node $v_t$, we move to any neighbor of $v_t$ with equal probability. The $t$-step random walk distribution $P_t$ of $v_0$ is 

$$
P_t(i) = \text{Prob}(v_t=i)
$$

### Model Analysis
With a randomization assumption of the ReLU, we can draw connections between GCNs and random walks. 

> (Theorem 1) Given a $k$-layer GCN, assume that all paths in the computation graph of the model are activated with the same probability of success $\rho$. Then the influence distribution $I_x$ for any node $x\in V$ is equivalent to the $k$-step random walk distribution of $G$ starting at node $x$. 


To better understand the implication above and the limitation of the corresponding neighborhood aggregation algorithms, we revisit the scenario of learning on a social network. 

- Random walks starting inside an expander converge rapidly in $O(\log \|V\|)$ steps to an almost-uniform distribution. After $O(\log \|V\|)$ iterations of neighborhood aggression, the representation of every node is influenced almost equally by any other node in the expander. Therefore, the node representations will be representative of the global graph and carry limited information about individual nodes. 
- In contrast, random walks starting at the bounded tree-width part converge slowly, i.e., the features retain more local information. Models that impose a fixed random walk distribution inherit these discrepancies in the speed of expansion and influence neighborhoods, which may not lead to the best representation for all nodes. 

### Jumping Knowledge Networks
Large influence radius may lead to too much averaging, while small influence radius may lead to instabilities or insufficient information aggregation. To overcome the limitations, a 4-layer JK-Net is proposed.

As in common neighborhood aggregation networks, each layer increases the size of influence distribution by aggregating neighborhood from the previous layer. At the last layer, for each node, JK-Net carefully select from all of those intermediate representations. If this is done independently for each node, then the model can adapt the effective neighborhood size for each node as needed, resulting in exactly the desired adaptivity. 

![jknet framework]({{'/assets/images/jk-net.png'|relative_url}})

The key idea for the design of layer-aggregation function is to determine the importance of a node's subgraph features at different ranges after looking at the learned features on all layers, rather than to optimize and fix the same weight for all nodes. We show below that layer-wise max-pooling implicity learns the influence locality adaptively for different nodes.

> (Proposition 1) Assume that paths of the same length in the computation graph are activated with the same probability. The influence score $I(x,y)$ for any $x,y\in V$ under a $k$-layer JK-Net with layer-wise max-pooling is equivalent in expectation to a mixture of $0,...,k$-step random walk distributions on $G$ at $y$ starting at $x$, the coefficients of which depend on the values of the layer features $h_x^{(l)}$.

For a node affiliate to a hub, JK-Net learns to put most influence on the node itself and otherwise spreads out the influence. However, GCNs would not capture the importance of the node's own features in such a structure because the probability at an affiliate node is small after a few random walk steps.

For hubs, JK-Net spreads out the influence across the neighboring nodes in a reasonable range, which makes sense because the nodes connected to  the hubs are presumably as informative as the hubs' own features. 

### Intermediate Layer Aggregation and Structures

As we shall see, models with concatenation aggregation perform well on graphs with more regular structures such as images and well-structured communities. As a more general framework, JK-Net admits general layer-wise aggregation models and enables better structure-aware representations on graphs with complex structures.

## Proof

### Proof of Theorem 1
Here we show the influence distribution $I_x$ of $k$-step random walk is the same as GCNs with $k$-layers.
Define by $f_x^{(l)}$ the pre-activated feature of $h_x^{(l)}$.
Given the aggregation function of GCNs as $h_x^{(l)} = \frac{1}{\text{deg}(x)} \cdot \sum_{z \in N(x)} W_l h_z^{(l-1)}$, for any $l=1,...,k$, we have 

$$
\frac{\partial h_x^{(l)}}{\partial h_y^{(0)}} = \frac{1}{\text{deg}(x)}\cdot \text{diag}\big(1_{f_x^{(l)}>0}\big) \cdot W_l \cdot \frac{\partial h_x^{(l-1)}}{\partial h_y^{(0)}}
$$

By chain rule, we get 

$$
\frac{\partial h_x^{(k)}}{\partial h_y^{(0)}} = \sum_{p=1}^\Psi \big[\frac{\partial h_x^{(k)}}{\partial h_y^{(0)}} \big]_p = \sum_{p=1}^\Psi \prod_{l=k}^1 \frac{1}{\text{deg}(v_p^l)} \text{diag}\big(1_{f_{v_p^l}^{(l)}>0}\big)\cdot W_l
$$

Here, $\Psi$ is the total number of paths $v_p^k, v_p^{k-1}, ..., v_p^0$ of length $k+1$ from node $x$ to node $y$. For any path $p$, $v_p^k$ is node $x$, and $v_p^0$ is node $y$ for $l=1,...,k-1$, and $v_p^{l-1} \in N(v_p^l)$.

As for each path $p$, the derivative $\big[ \frac{\partial h_x^{(k)}}{\partial h_y^{(0)}} \big]_p$ represents a directed acyclic computation graph, where the input neurons are the same as the entries of $W_1$, and at a layer $l$. We can express an entry of the derivative as

$$
\big[ \frac{\partial h_x^{(k)}}{\partial h_y^{(0)}} \big]_p^{(i,j)} = \prod_{l=k}^1 \frac{1}{\text{deg}(v^l_p)} \sum_{q=1}^\Phi Z_q \sum_{l=k}^1 w_q^{(l)}
$$

Here, $\Phi$ is the number of paths $q$ from the input neurons to the output neuron $(i,j)$, in the computation graph of $\big[ \frac{\partial h_x^{(k)}}{\partial h_y^{(0)}} \big]$. For each layer $l$, $w_q^l$ is the entry of $W_l$ that is used in the $q$-th path. Finally, $Z_q \in \{ 0,1 \}$ represents whether the $q$-th path is activate $Z_q=1$ or not $Z_q=0$ as a result of the ReLU activation of the entries of $f_{v_p^l}^{(l)}$'s on the $q$-th path.

Under the assumption that the $Z$'s are Bernoulli random variables with the same probability of success, for all $q$, $\text{Prob}(Z_q=1)=\rho$, we have 

$$
\mathbb{E}\Big[ \big[ \frac{\partial h_x^{(k)}}{\partial h_y^{(0)}} \big]_p^{(i,j)} \Big] = \rho \cdot \prod_{l=k}^1 \frac{1}{\text{deg}(v_l^p)} \cdot w_q^{(l)}
$$

It follows that 

$$
\mathbb{E}\big[ \frac{\partial h_x^{(k)}}{\partial h_y^{(0)}} \big] = \rho\cdot\prod_{l=k}^1 W_l \cdot \big(\sum_{p=1}^\Psi \prod_{l=k}^1 \frac{1}{\text{deg}(v_p^l)} \big)
$$

We know that the $k$-step random walk probability at $y$ can be computed by summing up the probability of all paths of length $k$ from $x$ to $y$, which is exactly $\sum_{p=1}^\Psi \prod_{l=k}^1 \frac{1}{\text{deg}(v_p^l)}$. Moreover, the random walk probability starting at $x$ to other nodes sum up to $1$. 

We know that the influence score $I(x,z)$ for any $z$ in expectation is thus the random walk probability of being at $z$ from $x$ at the $k$-th step, multiplied by a term that is the same for all $z$. Normalizing the influence scores ends the proof.