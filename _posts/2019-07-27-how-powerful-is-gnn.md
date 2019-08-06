---
layout: post
comments: false
title: "How Powerful are Graph Neural Networks"
categories: misc
---

> Abstract: This work characterizes the discriminative power of graph neural networks (GNNs), and show that GNNs cannot learn to distinguish certain simple graph structures. Then, a simple architecture is developed to distinguish certain simple graph structures for a graph classification task.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Introduction
This work formally characterizes how expressive different Graph Neural Networks (GNNs) are in learning to represent the distinction between different graph structures. The framework is inspired by the close connection between GNNs and the Weisfeiler-Lehman (WL) graph isomorphism test. Similar to GNNs, the WL test iteratively updates a given node's feature vector by aggregating feature vectors of its network neighbors. 

What makes WL test powerful is its injective aggregation update that maps different node neighborhoods to different feature vectors. We represent the set of feature vectors of a node as a multiset. Then, the neighbor aggregation in GNNs can be thought of as an aggregation function over the multisets into different representations.

![WL framework]({{'/assets/images/wl_test.png'|relative_url}})


## Preliminaries
Let $G=(V,E)$ denote a graph with node feature vector $X_v$ for $v\in V$. We are interested in two tasks:

- Node classification $y_v \in f(h_v)$
- Graph classification $g_G = g(h_G)$

### Graph Neural Networks
We define the $k$-th layer of GNN as 

$$
\begin{aligned}
 a_v^{(k)} &= \text{aggregate}^{(k)}\left(\{h_u^{(u-1)}: u \in N(v)\}\right) \\
 h_v^{(k)} &= \text{combine}^{(k)}\left(h_v^{(k-1)}, a_v^{(k)}\right)
\end{aligned}
$$

For graph classification, "readout" function is required to aggregates node feature from the final iteration to obtain the entire graph's representation $h_G$:

$$
h_G = \text{readout}\left(\{h_v^{(k)} | v\in G\}\right)
$$

### Weisfeiler-Lehman Test
The graph isomorphism problems ask whether two graphs are topologically identical.
Apart from some corner cases, WL test of graph isomorphism is an effective and computationally efficient test that distinguishes graphs. 

The WL test iteratively

- **aggregates** the labels of nodes and their neighborhoods
- **hashes** the aggregated labels into unique new labels

## Theoretical Framework
![Theoretical framework]({{'/assets/images/theoretical_framework.png'|relative_url}})

To study the representation power of a GNN, we analyze when a GNN maps two nodes to the same location in the embedding space. Intuitively, two nodes are mapped to the same location only if they have **identical subtree structures** with **identical features** on the corresponding nodes. 

This problem can be reduced as analyzing whether a GNN maps two neighborhoods (multisets) to the same embedding. A powerful GNN will never map two different neighborhoods to the same representation. This means the aggregation scheme should be **injective**.

Therefore, we abstract a GNN's aggregation scheme as a class of functions over multisets that their neural networks can represent and analyze whether they can represent injective multiset function.

## Building Powerful GNNs
The ability to map any two different graphs to different embeddings implies solving the challenging **graph isomorphism problem**. That is, we want isomorphic graphs to be mapped to the same representation and non-isomorphic ones to different representations.

> (Lemma 1) Let $G_1$ and $G_2$ be any two non-isomorphic graphs. If a GNN $\mathcal{A}: \mathcal{G}\rightarrow \mathbb{R}^d$ maps $G_1$ and $G_2$ to different embeddings, the WL test also decides $G_1$ and $G_2$ are not isomorphic.

The lemma above is correct by showing for an arbitrary iteration $i$ and node $u,v \in V$, that $l_u^{(i+1)}=l_v^{(i+1)}$ implies $h_u^{(i+1)}= h_v^{(i+1)}$

> (Theorem 2) With a sufficient number of GNN layers, a GNN $\mathcal{A}: \mathcal{G}\rightarrow \mathbb{R}^d$ maps any graphs $G_1$ and $G_2$, which WL test decides as non-isomorphic, to different embeddings if the following conditions hold $h_v^{(k)} = \phi\left(h_v^{(k-1)}, f\left(\{h_u^{(k-1)}:u \in N(v)\}\right)\right)$, where $\phi$ and $f$ are injective, and graph-level readout is injective.

The theorem above is correct by showing that for an arbitrary iteration $i$, there exists an injective function $\phi$ that $h_v^{(i)} = \phi(l_v^{(i)})$

A GNN satisfies above theorem not only discriminate different structures but also map similar graph structures to similarity embeddings and capture dependencies between graph structure.

## Graph Isomorphism Network

For countable sets, injectiveness well characterizes whether a function preserves the distinctness of inputs. In the next lemma, we state that sum aggregators can represent injective universal functions over multisets.

> (Lemma 3) 
> Assume $\mathcal{X}$ is countable. There exists a function $f: \mathcal{X} \rightarrow \mathbb{R}^n$ so that $h(X) = \sum_{x\in X} f(x)$ is unique for each multiset $X \subset \mathcal{X}$ of bounded size. Moreover, any multiset function $g$ can be decomposed as $g(X)=\phi(\sum_{x\in X} f(x))$ for some function $\phi$.

The correctness of lemma can be shown by proving there exists a mapping $f$ such that $\sum_{x\in X} f(x)$ is unique.

The next corollary provides a simple and concrete formulation among many such aggregation schemes.

> (Corollary 4) 
> Assume $\mathcal{X}$ is countable. There exists a function $f: \mathcal{X} \rightarrow \mathbb{R}^n$ so that for infinitely many choices of $\epsilon$, $h(c,X) = (1+\epsilon)\cdot f(c) + \sum_{x\in X} f(x)$ is unique for each pair $(c, X)$, where $c \in \mathcal{X}$ and $X \subset \mathcal{X}$ is a multiset of bounded size. Moreover, any function $g$ over such pairs can be decomposed as $g(c,X)=\phi((1+\epsilon)\cdot f(c) + \sum_{x\in X} f(x))$ for some function $\phi$.

We can use neural networks to model $f$ and $\phi$. The Graph Isomorphism Network (GIN) updates node representations as

$$
h_v^{(k)} = \text{MLP}^{(k)}\left((1+\epsilon^{(k)})\cdot h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)}\right)
$$

Readout function

$$
\begin{aligned}
h_G^{(k)} &= \text{readout}\left(h_v^{(k)}|v \in G\right) \\
h_G &= \text{concat}\left(h_G^{(k)}|k=0,1,...,K\right)
\end{aligned}
$$

## Less powerful but still interesting GNNs

### 1-Layer perceptrons are not sufficient
The following lemma shows that there are indeed network neighborhoods that models with 1-layer perceptrons can never distinguish.
> (Lemma 5)
> There exist finite multisets $X_1 \neq X_2$ so that for any linear mapping $W$, $\sum_{x \in X_1} \text{ReLU}(Wx) = \sum_{x \in X_2} \text{ReLU}(Wx)$

The lemma is correct because 1-layer perceptions can behave much like linear mappings, so that GNN layers degenerate into simply summing over neighborhood features.

### Structures that confuse mean and max-pooling
The mean and max operations are not injective. 
![Expression Power]({{'/assets/images/expression_power.png'|relative_url}})
![Fail cases]({{'/assets/images/fail_cases.png'|relative_url}})

### Mean learns distributions
The mean operation captures the distribution (proportion) of elements in a multiset, but not the exact multiset. The mean aggregator may perform well if the statistical and distributional information in the graph is more important than the exact structure.

> (Corollary 6) Assume $\mathcal{X}$ is countable. There exists a function $f: \mathcal{X}\rightarrow\mathbb{R}^n$ so that for $h(X) = \frac{1}{\|X\|}\sum_{x\in X}f(x)$, $h(X_1)=h(X_2)$ if and only if multisets $X_1$ and $X_2$ have the same distribution. That is, assuming $\|X_2\| \geq \|X_1\|$, we have $X_1 = (S,m)$ and $X_2=(S,k\cdot m)$ for some $k\in \mathbb{N}_{\geq 1}$.

### Max learns sets with distinct elements
Max-pooling captures neither the exact structure nor the distribution. However, it may be suitable for tasks where it is important to identify representative elements or the "skeleton", rather than to distinguish the exact structure or distribution.

> (Corollary 7) 
> Assume $\mathcal{X}$ is countable. Then there exists a function $f:\mathcal{X}\rightarrow \mathbb{R}^\infty$ so that for $h(X)=\max_{x\in X} f(x), h(X_1)=h(X_2)$ if and only if $X_1$ and $X_2$ have the same underlying set.

![performance]({{'/assets/images/gin_performance.png'|relative_url}})

## Future directions

1. This paper investigates **representational power** of GNNs. However, **generalization**, **inductive bias** and **optimization** of different GNN variants are still not explored.
2. This paper develops a theoretical framework for **graph classification** by analyzing the representation power of GNN. However, a theoretical framework for **graph regression**, **node classification** [3] and **edge classification** are still unexplored.
3. In this paper, the authors also try to concatenate a node with its neighbors. Interestingly, such concatenation was harder to train compared to our simple GINs (both GIN-0 and GIN-eps) and achieved lower test accuracy. 
4. Ideas from **graph minor theory** [4] and **spectral graph theory** [5] are not fully explored in the current message passing frameworks
5. For countable set, injectiveness well characterizes whether a function preserves the distinctness of inputs. Uncountable sets, where node features are continuous need some further considerations. Besides, it would be interesting to characterize how close together the learned features lie in a function's image. measure-theoretic analysis.
6. Apply GIN to analyze the understand other aggregation schemes.
7. Why GIN-$\epsilon$ is less powerful than GIN-$0$ while testing?
8. Understand and improve the generalization properties of GNNs as well as better understand their optimization landscape.
9. The paper shows that GNNs are at most as powerful as $1$-WL for representation. However, $1$-WL might fail under several circumstances. We can investigate combining GNN with $k$-WL [2].
10. The shortcomings of neighbor aggregation procedures are not considered. In particular, the "neighborhood" nodes that a node's representation draws from depends on the graph structure.
11. "sum" operation is not always injective. One can have $f_1 + f_2 = f_3 + f_4$ even though $f_1 \neq f_2 \neq f_3 \neq f_4$. 
12. The neighbor of each node is various a lot. "sum" operation is not the best choice. 
13. The performance improvement is minor.
14. The author oversimplifies baseline models. For example, GCNs use a normalized sum aggregation, which is injective.
15. Large graph? Random sampling and random subtree selection is obviously not good because of the injectiveness.
    
## Reference

[1] Xu, Keyulu, et al. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).

[2] Morris, Christopher, et al. "Weisfeiler and leman go neural: Higher-order graph neural networks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. 2019.

[3] Li, Qimai, Zhichao Han, and Xiao-Ming Wu. "Deeper insights into graph convolutional networks for semi-supervised learning." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

[4] https://www.birs.ca/workshops/2008/08w5079/report08w5079.pdf

[5] http://www.cs.yale.edu/homes/spielman/561/