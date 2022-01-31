# Linear Regression


# Steps to solving a supervised learning problem
- Collect the data set (input-output pairs)
- Choose a class of hypotheses (hypothesis space) $\mathcal{H}$
- Choose a hypothesis $h \in \mathcal{H}$

# Problem formulation

- Let $\mathcal{X}$ denote the space of input values.
- Let $\mathcal{Y}$ denote the space of output values.
- Given a dataset $S \in \mathcal{X} \times \mathcal{Y}$, find a fuction such that:

$$
h : \mathcal{X} \rightarrow \mathcal{Y}
$$

# Linear Model:

- In linear regression, we consider the model $h_w$ has the form:
    $$
    h_w(x) = \textbf{w}^T x + b
    $$
    where $\textbf{w} = [w_1, \cdots, w_2]^T \in \mathbb{R}^n$.

- We will choose $\textbf{w}$ such that the error function is minimized:
  $$
   w = \argmin_w \sum_{i = 1}^m l(h_w(x_i), y_i)
  $$
# Least square: 

We use the squared error to measure our prediction performance, and choose $w$ by minimizing the sum-of-squared errors:
  $$
  J(w) = \frac{1}{2}\sum_{i = 1}^m(h_w(x_i) - y_i)^2
    $$
To minimize $J(w)$, we need to solve $\nabla J(w) = 0$

$$
    \nabla J(w) = [\frac{\partial J(w)}{\partial w_1} , \cdots , \frac{\partial J(w)}{\partial w_n}]^T
$$
where:
$$
    \begin{aligned}
        \frac{\partial J(w)}{\partial w_j} &= 
        \frac{\partial}{\partial w_j}  \frac{1}{2}\sum_{i = 1}^m (h_w(x_i) - y_i)^2\\
        &= \frac{\partial}{\partial w_j}  \frac{1}{2}\sum_{i = 1}^m(\textbf{w}^T x_i + b - y_i)^2\\
        & =  \frac{1}{2} \cdot 2 \sum_{i = 1}^m (\textbf{w}^T x_i + b - y_i)\frac{\partial}{\partial w_j} (\textbf{w}^T x_i + b - y_i)\\
        & = \sum_{i = 1}^m (\textbf{w}^T x_i + b - y_i)\frac{\partial}{\partial w_j}x_{i,j}

    \end{aligned}
$$

if we consider the vector form:

$$
    \begin{aligned}
    \nabla J(w) &= \nabla _w (\frac{1}{2} \sum_{i = 1}^m (h_w(x_i) - y_i)^2)\\
    &= \nabla _w (\frac{1}{2} (Xw - y)^T(Xw - y))\\
    &= \nabla _w (\frac{1}{2} (w^TX^TXw - y^TXw - w^TX^Ty + y^Ty)\\
    &= X^TXw - X^Ty
    \end{aligned}
$$
And we setting $\nabla J(w) = 0$, we get:
$$
    \begin{aligned}
        X^TXw - X^Ty &= 0\\
        w &= (X^TX)^{-1}X^Ty
    \end{aligned}
$$

# Probabilistic perspective of linear regression

Assume that there exist $w$ such that:

$$
y_i = h_w(x_i) + \epsilon_i
$$

where

$$
\epsilon_i \backsim N(0,\sigma^2)
$$


## Bayes Rule

$$
P(h|D) = \frac{P(D|h)P(h)}{P(D)}
$$

- $P(h)$ is the prior probability of hypothesis $h$
- $P(D) = \int_h P(D|h)P(h)$ is the probability of training data $D$
- $P(h|D)$ is the probability of $h$ given $D$
- $P(D|h)$ is the probability of $D$ given $h$ (likelihood of the data)


## Maximum Likelihood estimation
Since we have:

$$
y = h_w(x) + \epsilon, \text{where } \epsilon \backsim N(0,\sigma^2)
$$
Hence,

$$
y|x;h \backsim N(h_w(x),\sigma^2)
$$

And we can write:

$$
P(y|x; h) = \frac{1}{\sqrt{2\pi}\sigma} \exp^{- \frac{1}{2}\frac{y - h_w(x)^2}{\sigma^2}}
$$

And we want to find the maximum value :
$$
\begin{aligned}
    \log L(h) &= \log P(Y | X; h) \\
    &= \log \prod_{i = 1}^n P(y_i|x_i;h)\\
    &= \sum_{i = 1}^n \log P(y_i|x_i;h)\\
    &= \sum_{i = 1}^n \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2}\frac{y_i - h_w(x_i)^2}{\sigma^2}
\end{aligned}
$$

Maximizing this with respect to $h$, we can find a optimal $h$:
$$
\begin{aligned}
h &= \argmax_h L(h)\\
&=\argmin_h \frac{1}{2}\frac{y_i - h_w(x_i)^2}{\sigma^2}
\end{aligned}
$$

Which again is the sum-square-error function. This means sum-square-error also assume the noise follows the Gaussian distribution.

## Maximum a posteriori(MAP) hypotheshis $h_{MAP}$

$$
\begin{aligned}
    h_{MAP} &= \argmax_{h \in \mathcal{H}} P(h|D)\\
    &= \argmax_{h \in \mathcal{H}} \frac{P(D|h)P(h)}{P(D)}
\end{aligned}
$$

Last step is $P(D)$ because $P(D)$ is independent of $h$ (so constant for the maximization).
