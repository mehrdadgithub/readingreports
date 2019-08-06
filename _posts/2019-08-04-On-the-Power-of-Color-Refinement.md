---
layout: post
comments: false
title: "On the Power of Color Refinement"
categories: misc
---

> Abstract: This work aims at determining the exact range of applicability of color refinement algorithms.



<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Introduction
 Color refinement is a classical technique used to show that two given graphs G and H are non-isomorphic; it is very efficient, although it does not succeed on all graphs. We call a graph $G$ amenable to color refinement if the color-refinement procedure succeeds in distinguishing $G$ from any non-isomorphic graph $H$. 
 
 In this work, the authors determine the exact range of applicability of color refinement by showing that amenable graphs are recognizable in time $O((n + m) \log n)$, where $n$ and $m$ denote the number of vertices and the number of edges in the input graph.

Review that the well-known color refinement (also known as 1-WL test) procedure for Graph Isomorphism works as follows: 

- It begins with a uniform coloring of the vertices of two graphs $G$ and $H$ and refines the vertex coloring step by step. 
- In a refinement step, if two vertices have identical colors but differently colored neighborhoods (with the multiplicities of colors counted), then these vertices get new different colors. 
- The procedure terminates when no further refinement of the vertex color classes is possible. Upon termination, if the multisets of vertex colors in $G$ and $H$ are different, we can correctly conclude that they are not isomorphic.

Color refinement sometimes fails to distinguish non-isomorphic graphs. The simplest example is given by any two non-isomorphic regular graphs of the same degree with the same number of vertices. According to the existing works, there are interesting classes of amenable graphs:

- Unigraphs, i.e., graphs that are determined up to isomorphism by their degree sequences
- Trees
- Graphs for which the color refinement procedure terminates with all singleton color classes, i.e. the color classes form the discrete partition

## Notations

 A set of vertices $X \subseteq V (G)$ induces a subgraph of $G$, that is denoted by $G[X]$.
 For two disjoint sets $X$ and $Y$ , $G[X, Y]$ is the bipartite graph with vertex classes $X$ and $Y$ formed by all edges of $G$ connecting a vertex in $X$ with a vertex in $Y$

Given a graph $G$, the color-refinement algorithm iteratively computes a sequence of colorings $C^i$ of $V (G)$. The initial coloring $C^0$ is the vertex coloring of $G$, 
Then, 

$$
C^{i+1}(u) = ( C^i(u), \{C^i(a) : a \in N(u)\} )
$$

The partition $\mathcal{P}^{i+1}$ of $V(G)$ into the color classes of $C^{i+1}$ is a refinement of the partition $P^i$ corresponding to $C^i$. It follows that, eventually, $\mathcal{P}^{s+1} = \mathcal{P}^s$ for some $s$; hence, $\mathcal{P}^i = \mathcal{P}^s$ for all $i \geq s$. The partition $\mathcal{P}^s$ is called the stable partition of $G$ and denoted by $\mathcal{P}_G$.


## Reference

