---
layout: post
comments: false
title: "Convex Optimization"
categories: misc
---


> Abstract: This blog reviews some gradient descent algorithms under convex assumptions.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Black-box model
In the black-box model we assume that we have unlimited computational resources, the set of constraint $\mathcal{X}$ is known, and the objective function $f: \mathcal{X}\rightarrow \mathbb{R}$ is unknown but can be accessed through queries to **oracles**:
- the zeroth-order oracle takes as input a point $x\in\mathcal{X}$ and outputs the value of $f$ at $x$
- the first-order oracle takes as input a point $x\in\mathcal{X}$ and outputs a subgradient of $f$ as $x$

We are interested in the oracle complexity (upper bound) of convex optimization: how many queries to the oracles are necessary to find a $\varepsilon$-approximate minima of a convex function

## Dimension-free convex optimization

We define projection operator $\Pi_\mathcal{X}$ on $\mathcal{X}$ as $\Pi_\mathcal{X}(x) = \arg\min_{y\in\mathcal{X}} \|x-y\|$. 

### Projected subgradient descent for Lipschitz function
The projected subgradient descent algorithm iterates the following equations for:
$$
\begin{aligned}
&y_{t+1} = x_t - \eta g_t, \text{where} ~g_t \in \partial f(x_t)
\\
&x_{t+1} = \Pi_\mathcal{X}(y_{t+1})
\end{aligned}
$$

> **Theorem 1.1.** The projected subgradient descent method with $\eta = \frac{R}{L\sqrt{t}}$ satisfies 
> $$
> f\left(\frac{1}{t}\sum_{s=1}^t x_s\right) - f(x^*) \leq \frac{RL}{\sqrt{t}}
> $$

**Proof of Theorem 1.1.** Using the definition of subgradient and the elementary identity $2a^\top b = \|a\|^2+\|b\|^2-\|a-b\|^2$ one obtains

$$

\begin{aligned}
f(x_s)-f(x^*) &\leq g_s^\top(x_s - x^*) \\
&= \underbrace{\frac{1}{\eta}(x_s-y_{s+1})^\top}_{y_{t+1} = x_t - \eta g_t}(x_s - x^*) \\
&= \frac{1}{2\eta} \left( \underbrace{\|x_s-x^*\|^2 + \|x_s-y_{s+1}\|^2 - \|y_{s+1}-x^*\|^2}_{2a^\top b = \|a\|^2+\|b\|^2-\|a-b\|^2}\right) \\
&= \frac{1}{2\eta} \left( \|x_s-x^*\|^2 - \|y_{s+1}-x^*\|^2\right) + \underbrace{\frac{\eta}{2}\|g_s\|^2}_{y_{t+1} = x_t - \eta g_t}
\end{aligned}
$$

Note that function is $L$-Lipschitz continuos and 

$$\|y_{s+1}-x^*\|\geq \|x_{s+1}-x^*\|$$

Summing the resulting inequality over $s$ and using that $\| x_1-x^* \| \leq R$ yield

$$
\sum_{s=1}^t \left( f(x_s)-f(x^*)\right) \leq \frac{R^2}{2\eta} + \frac{\eta L^2 t}{2}
$$

Plugging in the value of $\eta$ directly gives the statement.

<!---
We can show later that the rate is given in **Theorem 1** is unimprovable from a black-box perspective. 
--->
Thus to reach a $\varepsilon$-optimal point one needs $O(1/\varepsilon^2)$ calls to the oracle.

The computational bottleneck of the projected subgradient descent is the projection step, which is a convex optimization problem by itself. Later we will introduce a projection-free algorithm.

### Gradient descent for smooth functions
We say a continuously differentiable function $f$ is $\beta$-smooth if the gradient $\nabla f$ is $\beta$-Lipschitz:
$$
\|\nabla f(x)-\nabla f(y)\| \leq \beta \|x-y\|
$$

Here we introduce some properties of smooth convex functions. For any $x,y\in\mathbb{R}^n$, one has 
$$
\begin{aligned}
&0 \leq f(x)-f(y)-\nabla f(y)^\top (x-y) \leq \frac{\beta}{2}\|x-y\|^2 \\
&f(x)-f(y) \leq \nabla f(x)^\top (x-y) - \frac{\beta}{2}\|\nabla f(x)-\nabla f(y)\|^2
\end{aligned}
$$

The improvement in one step of gradient descent 
$$f\left(x-\frac{1}{\beta}\nabla f(x)\right) - f(x) \leq -\frac{1}{2\beta}\|\nabla f(x)\|^2$$

<hr>

**Theorem 1.2.** Let $f$ be convex and $\beta$-smooth on $\mathbb{R}^n$. The gradient descent with $\eta=\frac{1}{\beta}$ satisfies
$$
f(x_t)-f(x^*) \leq \frac{2\beta \|x_1-x^*\|^2}{t-1}
$$

**Proof of Theorem 1.2.** Because the improvement in one step of gradient descent is bounded as 
$$
f(x_{s+1}) - f(x_s) \leq -\frac{1}{2\beta}\|\nabla f(x_s)\|^2
$$

Denoting $\delta_s = f(x_s) - f(x^*)$, we have
$$
\delta_{s+1} \leq \delta_s - \frac{1}{2\beta}\|\nabla f(x_s)\|^2
$$

One also has by convexity 
$$
\delta_s \leq \nabla f(x_s)^\top (x_s-x^*) \leq \|x_s-x^*\|\cdot\|\nabla f(x_s)\|
$$


If we can show $\| x_s - x^* \|$ is decreasing with $s$, which with the two above displays will imply
$$
\delta_{s+1} \leq \delta_s - \frac{1}{2\beta \|x_1-x^*\|^2} \delta^2_s
$$

Let us set $w = \frac{1}{2\beta \|x_1-x^*\|^2}$, the above equation becomes 
$$
\begin{aligned}
&\delta_{s+1} \leq \delta_s - w \delta^2_s \\
\Leftrightarrow~ &w \frac{\delta_s}{\delta_{s+1}} + \frac{1}{\delta_s} \leq \frac{1}{\delta_{s+1}} \\
\Rightarrow~ &\frac{1}{\delta_{s+1}}- \frac{1}{\delta_{s}} \geq w &\delta_s/\delta_{s+1} \geq 1\\
\Rightarrow~ &\frac{1}{\delta_t} \geq w(t-1) &\text{telescope sum}
\end{aligned}
$$

Then we show $\| x_s - x^* \|$ is decreasing with $s$. 
$$
\begin{aligned}
\| x_{s+1}-x^*\|^2 &= \|x_s - \frac{1}{\beta} \nabla f(x_s) - x^* \|^2 \\
&= \|x_s - x^*\|^2 - \frac{2}{\beta} \nabla f(x_s)^\top (x_s - x^*) + \frac{1}{\beta^2} \|\nabla f(x_s)\|^2 \\
&\leq \|x_s - x^*\|^2 - \frac{1}{\beta^2} \|\nabla f(x_s)\|^2\\
&\leq \|x_s - x^*\|^2
\end{aligned}
$$

<hr>

### Conditional gradient descent (Frank-Wolfe)
The conditional gradient descent performs the following update 
$$
\begin{aligned}
&y_t \in \arg\min_{y\in\mathcal{X}} \nabla f(x_t)^\top y \\
&x_{t+1} = (1-\gamma_t) x_t + \gamma_t y_t
\end{aligned}
$$

A major advantage of conditional gradient descent over projected gradient descent is that the former can adapt to smoothness in an arbitrary norm.

<hr>

**Theorem 1.3.** Let $f$ be a convex $\beta$-smooth function w.r.t. some norm $\|\cdot\|$, $R=\sup_{x,y\in\mathcal{X}}\|x-y\|$, and $\gamma_s = \frac{2}{s+1}$ for $s\geq 1$. Then one has
$$
f(x_t) - f(x^*) \leq \frac{2\beta R^2}{t+1}
$$

**Proof of Theorem 1.3.** 
$$
\begin{aligned}
f(x_{s+1})-f(x_s) &\leq \nabla f(x_s)^\top (x_{s+1}-x_s) + \frac{\beta}{2}\|x_{s+1}-x_s\|^2 &\beta-\text{smooth}\\
&\leq \gamma_s \nabla f(x_s)^\top (y_s-x_s) + \frac{\beta}{2} \gamma^2_s R^2 &  x_{t+1} = (1-\gamma_t) x_t + \gamma_t y_t\\
&\leq \gamma_s \nabla f(x_s)^\top (x^*-x_s) + \frac{\beta}{2} \gamma^2_s R^2  & y_t \in \arg\min_{y\in\mathcal{X}} \nabla f(x_t)^\top y\\
&\leq \gamma_s (f(x^*)-f(x_s)) + \frac{\beta}{2} \gamma^2_s R^2
\end{aligned}
$$

Rewriting this inequality in terms of $\delta_s = f(x_s)-f(x^*)$ one obtains 
$$\delta_{s+1} \leq (1-\gamma_s)\delta_s + \frac{\beta}{2} \gamma^2_s R^2$$

Therefore $\gamma_s = \frac{2}{s+1}$ finishes the proof.
<hr>

In addition to being projection-free and norm free, the conditional gradient descent produces **sparse** iterations. Let consider the situation where $\mathcal{X} \subset \mathbb{R}^n$ is the convex hull of a finite set of points. Any point $x\in\mathcal{X}$ can be written as a convex combination of at most $n+1$ vertices of $\mathcal{X}$. On the other hand, by definition of the conditional gradient descent, one knows that the $t$-th iterate $x_t$ can be written as a convex combination of $t$ vertices. Since $t \ll n$, thus we see that the iterates of conditional gradient descent are very sparse in their representation.

## Strongly convexity
We say that function $f: \mathcal{X}\rightarrow \mathbb{R}$ is $\alpha$-strongly convex if it satisfy
$$
f(x) - f(y) \leq \nabla f(x)^\top (x-y) - \frac{\alpha}{2} \|x-y\|^2
$$

A function $f$ is $\alpha$-strongly convex if and only if $x\rightarrow f(x)-\frac{\alpha}{2}\|x\|^2$ is convex. 

We can say that any point $x$ one can find a quadratic lower bound $$q^-_x(y) = f(x)+\nabla f(x)^\top (y-x)+ \frac{\alpha}{2}\|x-y\|^2$$

On the other hand for $\beta$-smoothness implies that at any point $y$ can find a quadratic upper bound $$q^+_y(x) = f(y)+\nabla f(y)^\top (x-y) + \frac{\beta}{2}\|x-y\|^2$$

Therefore, we can say strong convexity if a dual assumption to smoothness.

### Strongly convex and Lipschitz functions
We consider the projected subgradient descent algorithm with time-varying step size $\eta_t$, that is 
$$
\begin{aligned}
&y_{t+1} = x_t - \eta_t g_t, \text{where} ~g_t \in \partial f(x_t)
\\
&x_{t+1} = \Pi_\mathcal{X}(y_{t+1})
\end{aligned}
$$

<hr>

**Theorem 1.4.** Let $f$ be $\alpha$-strongly convex and $L$-Lipschitz on $\mathcal{X}$. Then projected subgradient descent with $\eta_s = \frac{2}{\alpha (s+1)}$ satisfies $$f\left(\sum_{s=1}^t \frac{2s}{t(t+1)}x_s - f(x^*) \leq \frac{2L^2}{\alpha (t+1)}\right)$$

**Proof of Theorem 1.4.** 
$$
\begin{aligned}
f(x_s)-f(x^*) &\leq g_s^\top(x_s - x^*) - \frac{\alpha}{2}\|x_s-x^*\|^2\\
&= \underbrace{\frac{1}{\eta_s}(x_s-y_{s+1})^\top}_{y_{t+1} = x_t - \eta_s g_t}(x_s - x^*) - \frac{\alpha}{2}\|x_s-x^*\|^2\\
&= \frac{1}{2\eta_s} \left( \|x_s-x^*\|^2 + \|x_s-y_{s+1}\|^2 - \|y_{s+1}-x^*\|^2\right) - \frac{\alpha}{2}\|x_s-x^*\|^2\\
&\leq (\frac{1}{2\eta_s} - \frac{\alpha}{2})\|x_s-x^*\|^2 + \underbrace{\frac{1}{2\eta_s} \|x_s-y_{s+1}\|^2}_{x_t - y_{t+1} = \eta_t g_t \leq \eta_t L} - \underbrace{\frac{1}{2\eta_s} \|y_{s+1}-x^*\|^2}_{\|x_{s+1}-x^*\|^2 \leq \|y_{s+1}-x^*\|^2} \\
&\leq (\frac{1}{2\eta_s} - \frac{\alpha}{2})\|x_s-x^*\|^2 + \frac{\eta_s}{2} L^2 - \frac{1}{2\eta_s}\|x_{s+1}-x^*\|^2
\end{aligned}
$$

Multiplying this inequality by $s$ yields, and use $\eta_s = \frac{2}{\alpha (s+1)}$ one has
$$
\begin{aligned}
s(f(x_s)-f(x^*)) &\leq s(\frac{1}{2\eta_s} - \frac{\alpha}{2})\|x_s-x^*\|^2 + \frac{s \eta_s}{2} L^2 - \frac{s}{2\eta_s}\|x_{s+1}-x^*\|^2 &\\
&= \frac{L^2}{\alpha} + \frac{\alpha}{4} \left( s(s-1)\|x_s-x^*\|^2-s(s+1)\|x_{s+1}-x^*\|^2\right)\\
\end{aligned}
$$

By summing the resulting inequaltiy over $s=1$ to $s=t$, and apply Jensen's inequality to obtain the claimed statement.
<hr>

### Strongly convex and smooth functions
We say a function $f$ is $\alpha$-strongly convex and $\beta$-smooth if $$f(x)-f(y) \leq g_\mathcal{X}(x)^\top (x-y) - \frac{1}{2\beta}\|g_\mathcal{X}(x)\|^2 - \frac{\alpha}{2}\|x-y\|^2$$

<hr>

**Theorem 1.5.** Let $f$ be $\alpha$-strongly convex and $\beta$-smooth on $\mathcal{X}$. Then projected gradient descent with $\eta = 1/\beta$ satisfies $$\|x_{t+1}-x^*\|^2 \leq \exp\left(-\frac{t}{k}\right) \|x_1-x^*\|^2$$

**Proof of Theorem 1.5.** 
$$
\begin{aligned}
\|x_{t+1}-x^*\|^2 &= \|x_t - \frac{1}{\beta} g_\mathcal{X}(x_t) - x^*\|^2\\
&= -\frac{2}{\beta} \left( g_\mathcal{X}(x_t)^\top (x_t - x^*) - \frac{1}{2 \beta}\|g_\mathcal{X}(x_t)\|^2 -\frac{\beta}{2} \|x_t-x^*\|^2 \right) \\
&= -\frac{2}{\beta} \left( g_\mathcal{X}(x_t)^\top (x_t - x^*) - \frac{1}{2 \beta}\|g_\mathcal{X}(x_t)\|^2 -\frac{\alpha}{2} \|x_t-x^*\|^2 \right) + (1-\frac{\alpha}{\beta}) \|x_t-x^*\|^2\\
&\leq (1-\frac{\alpha}{\beta})^t \|x_1-x^*\|^2 \\
&\leq \exp\left(-\frac{t}{k}\right) \|x_1-x^*\|^2
\end{aligned}
$$

<hr>

Here we show that in the unconstrained case, one can improve the rate by a constant factor.

<hr>

**Theorem 1.5.** Let $f$ be $\alpha$-strongly convex and $\beta$-smooth on $\mathbb{R}^n$. Then gradient descent with $\eta = \frac{2}{\alpha+\beta}$ satisfies $$f(x_{t+1})-f(x^*) \leq \frac{\beta}{2} \exp \left(-\frac{4t}{\kappa+1} \right) \|x_1-x^*\|^2$$ 

<hr>

<!---
## Lower bound

## Geometric Descent
Let us define $B(x,r^2) := \{y\in\mathbb{R}^n: \|y-x\|^2 \leq r^2\}$ and 
$$
\begin{aligned}
x^+ &= x - \frac{1}{\beta} \nabla f(x) \\
x^{++} &= x - \frac{1}{\alpha}  \nabla f(x)
\end{aligned}
$$

Then we can rewrite the definition of strong convexity as
$$
\begin{aligned}
f(y) &\geq f(x) + \nabla f(x)^\top (y-x) + \frac{\alpha}{2}\|y-x\|^2 \\
\Leftrightarrow \frac{\alpha}{2}\|y-x-\frac{1}{\alpha} \nabla f(x)\|^2 &\leq \frac{\|\nabla f(x)\|^2}{2\alpha} - (f(x)-f(y))
\end{aligned}
$$

One obtains an enclosing ball for the minimizer of $f$ with the $0$-th and $1$-th order information at $x$:
$$
x^* \in B\left(x^{++}, \frac{\|\nabla f(x)\|^2}{\alpha^2} - \frac{2}{\alpha}(f(x)-f(y))\right)
$$

Furthermore, by smoothness one has $f(x^+)\leq f(x)-\frac{1}{2\beta}\|\nabla f(x)\|^2$ which allows to shrink the above ball by a factor of $1-\frac{1}{\kappa}$ and obtain the following:
$$
x^* \in B\left(x^{++}, \frac{\|\nabla f(x)\|^2}{\alpha^2}\left(1-\frac{1}{\kappa}\right) - \frac{2}{\alpha}(f(x^+)-f(x^*))\right)
$$

Assuming that we have an enclosing ball $A:=B(x, R^2)$ for $x$, we can then enclose $x$ in a ball $B$ containing the intersection of $B(x,R^2)$ and the ball $B(x^{++}, \frac{\|\nabla f(x)\|^2}{\alpha^2}(1-\frac{1}{\kappa}))$. Provided that the radius of $B$ is a fraction of the radius of $A$, one can then iterate the procedure by raplacing $A$ by $B$, leading to a linear convergence rate. 
--->
