---
layout: post
comments: false
title: "Weisfeiler-Lehman Graph Kernels"
categories: misc
---

> Abstract: This paper propose a family of efficient kernels for large graphs with discrete node labels based on the Weisfeiler-Lehman test of isomorphism on graphs. 

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Possibly the most natural measure of similarity of graphs is to check whether the graphs are topologically identical, that is, isomorphic. This gives rise to a binary similarity measure, which equals $1$ if the graphs are isomorphic, and $0$ otherwise. Despite the idea of checking graph isomorphism being so intuitive, no efficient algorithms are known for it. The graph isomorphism problem is in NP, but has been neither proven NP-complete nor found to be solved by a polynomial-time algorithm.

Recently, graph kernels are proposed to exploit graph topology, but restrict themselves to comparing substructures of graphs that are computable in polynomial time. Graph kernels bridge the gap between graph-structured data and a large spectrum of machine learning algorithms called kernel methods. A kernel is a function of two objects that quantifies their similarity. Mathematically, it corresponds to an inner product in a reproducing kernel Hilbert space

Several different graph kernels have been defined in machine learning which can be categorized into three classes: 

- graph kernels based on **walks** and **paths** 
- graph kernels based on limited-size **subgraphs**
- graph kernels based on **subtree** patterns

We define a graph as $G=(V,E,\ell)$ where $\ell: V \rightarrow \Sigma$ is a function that assign labels from an alphabet $\Sigma$ to nodes in the graph. For simplicity, we assume that every graph has $n$ nodes, $m$ edges, and a maximum degree of $d$. The size of $G$ is defined as the cardinality of $V$.

## The Weisfeiler-Lehman Test of Isomorphism
Graph kernels use concepts from the Weisfeiler-Lehman test of isomorphism, a.k.a. "naive vertex refinement". Assume we are given two graphs $G$ and $G'$ and we would like to test whether they are isomorphic. The key idea of the algorithm is to augment the node labels by the sorted set of node labels of neighboring nodes, and compress these augmented labels into new, short labels. These steps are then repeated until the node label sets of $G$ and $G'$ differ, or the number of iterations reaches $n$.

![WL framework]({{'/assets/images/WL-graph-kernel.png'|relative_url}})

## The General Weisfeiler-Lehman Kernels

### The Weisfeiler-Lehman Kernel Framework

We define the WL graph at height $i$ of the graph $G=(V,E,\ell)$ as the graph $G_i = (V,E,l_i)$. We call the sequence of WL graphs

$$
\{G_0,...,G_h\} = \{(V,E,l_0),...,(V,E,l_h)\}
$$

where $G_0 = G$ is the original graph and $l_0 = \ell$, the WL sequence up to height $h$ of $G$, and $G_1 = r(G_0)$ is the graph resulting from the first relabeling.

Let $k$ be any kernel for graphs, that we will call the base kernel. Then the WL kernel with $h$ iterations with the base kernel $k$ is defined as

$$
k_{WL}^{(h)}(G,G') = k(G_0,G_0^\prime) +  k(G_1,G_1^\prime) + ... + k(G_h,G_h^\prime)
$$

where $h$ is the number of WL iterations, and $\{G_0, ..., G_h\}$ and $\{G_0^\prime, ..., G_h^\prime\}$ are the WL sequences of $G$ and $G^\prime$ respectively.

> Let he base kernel $k$ be any Positive Semi-Definite (PSD) kernel on graphs. Then the corresponding WL kernel $k_{WL}^{(h)}$ is PSD.

This definition provides a framework for applying all graph kernels that take into account discrete node labels to different levels of the node-labeling of graphs, from the original labeling to more and more fine-grained labeling for growing $h$. 

In the inductive learning setting, we compute the kernel on the training set of graphs. For any test graph that we subsequently need to classify, we have to map it to the feature space spanned by original and compress labels occurred in the training set.

## Reference

