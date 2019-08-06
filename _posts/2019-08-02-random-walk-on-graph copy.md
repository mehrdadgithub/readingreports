---
layout: post
comments: false
title: "Random Walks on Graph"
categories: misc
---

> Abstract: In this blog various aspects of the theory of random walks on graphs are surveyed.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Basic Notations and Facts
Let $G=(V,E)$ be a connected graph with $n$ nodes and $m$ edges. Consider a random walk on $G$: we start at a node $v_0$; if at the $t$-th step we are at a node $v_t$, we move neighbor of $v_t$ with probability $1/d(v_t)$.
Clearly, the sequence of random nodes $(v_t: t=0,1,...)$ is a Markov chain.
We denote the distribution of $v_t$ by $P_t \in \mathbb{R}^{\|V\|}$ as 

$$
P_t(i) = \text{Prob}(v_t = i)
$$

We denote by $M = (p_{ij})$ the matrix of transition probabilities of this Markov chain

$$
p_{i,j} = \begin{cases}
1/d(i) & \text{ if } ij \in E \\ 
0 & \text{ otherwise }
\end{cases}
$$

Let $A$ be the adjacent matrix and let $D$ denote the diagonal matrix with $D_{ii} = 1/d(i)$, then $M = DA$. 
If $G$ is $d$-regular (every node has $d$ neighbors), then $M=(1/d)A$.
The rule of walk can be expressed by 

$$
P_{t+1} = M^\top P_t
$$

and hense

$$
P_t = (M^\top)^t P_0
$$

It follows that the probability $p_{ij}^t$ that starting at $i$ and reach at $j$ in $t$ steps is given by the $ij$-entry of the matrix $M^t$.

If $G$ is regular, then this Markov chain is **symmetric**: the probability of moving $u \rightarrow v$ is the same as the probability of moving $v \rightarrow u$.
If $G$ is non-regular, the Markov chain is **time-reversibility**: a random walk considered backwards is also a random walk. 

We say that the distribution $P_t$ is stationary if $P_{t+1}=P_t$. We say a walk is stationary walk if $P_t = P_0$.
In particular, the distribution on $V$ is stationary if the graph is regular. 
In addition, if graph $G$ is non-bipartite, the distribution of $v_t$ tends to a stationary distribution as $t \rightarrow \infty$.

## Main Parameters

The **access time** or **hitting time** $H_{ij}$ is the expected number of steps before node $j$ is visited, starting from $i$. 
The **commute time** is defined as the expected number of steps in a random walk starting at $i$, before node $j$ is visited and then node $i$ is reached again:

$$
\kappa(i,j) = H(i,j) + H(j,i)
$$

The **cover time** is the expected number of steps to reach every node. 
The **mixing rate** is a measure of how fast the random walk converges to its limiting distribution. For the non-bipartite graph, $p_{i,j}^{(t)}\rightarrow \frac{d(v_t)}{2m}$ as $t\rightarrow \infty$, and the mixing rate is 

$$
\mu = \lim_{t\rightarrow \infty} \max_{i,j} \Big|~p_{ij}^{(t)}- \frac{d(v_j)}{2m}~\Big|^{1/t}
$$

The **mixing time** is defined as the number of steps before the distribution of $v_t$ close to uniform. This number is about $\frac{\log n}{1-\mu}$. A surprising fast is that the mixing time is much less than the number of nodes, about $O(\log n)$ steps.

## Reference

