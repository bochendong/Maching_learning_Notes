# Markov Processes

# The Markov Property
- The Markov property means a process or state is memoryless: what happens next only depends on where you are right now.
- Future is independent of past given present.
- Information state: sufficient statistic of history.
- State $s_t$ is Markov if and only if:

$$
p(s_{t+1} | s_t) = p(s_{t+1} | s_1, s_2,\cdots, s_t)
$$

# The Markov Chain

A Markov Chain | Markov Process is a tuple $\mathcal{(S, P)}$, where:

- $\mathcal{S}$ is a finite set of states
- $\mathcal{P}$ is our state-transition probability matrix.
$$
P(S_{t+1} = s^{\prime}| S_t = s)
$$
- If finite number ($N$) of states, can express $P$ as a matrix:
    $$
    P = 
    \begin{bmatrix}
        p(s_1 | s_1) & p(s_2 | s_1) & \cdots & p(s_N | s_1)\\
        p(s_1 | s_2) & p(s_2 | s_2) & \cdots & p(s_N | s_2)\\
        \vdots       & \vdots       & \ddots & \vdots\\
        p(s_1 | s_N) & p(s_2 | s_N) & \cdots & p(s_N | s_N)
        \end{bmatrix}
    $$

Example:


<div align=center>
        <img src ="31.png" width="340" height ="250"/>
</div>


$$

P = 
\begin{bmatrix}
    0   & 0.8 & 0.2 & 0     \\
    0   & 0.3 & 0.7 & 0  \\
    0   & 0.9 & 0   & 0.1  \\
    0   & 0   & 0   & 1\\
\end{bmatrix}
$$

# Markov Reward Processes

A Markov Decision Process is a tuple $\mathcal{(S, P, R, \gamma)}$ where:

- $\mathcal{S}$ is a finite set of states.
- $\mathcal{P}$ is our state-transition probability matrix.
- $\mathcal{R}$ is a reward function, $R_s = \mathbb{E}[R_{t+1} | S_t = s]$
- $\mathcal{\gamma}$ is a discount factor $\gamma \in [0, 1]$
    - $\gamma = 0$: only care about immediate reward.
    - $\gamma = 1$: Future reward is as benefical as immediate reward.



# Markov Decision Processes (MDPS)

A Markov Decision Process is a tuple $\mathcal{(S, P, R, \gamma, A)}$  where:
- $\mathcal{S}$ is a finite set of states.
- $\mathcal{A}$ is a finite set of actions.
- $\mathcal{P}$ is our state-transition probability matrix.
$$
P(S_{t+1} = s^{\prime}| S_t = s, A_t = a)
$$
- $\mathcal{R}$ is a reward function, $R_s = \mathbb{E}[r_{t+1} | S_t = s, A_t = a]$
- $\mathcal{\gamma}$ is a discount factor $\gamma \in [0, 1]$
- $G$ is the discount sum of rewards from time step $t$ yo $H$

$$\begin{aligned}
    G_t &= r_{t + 1} + \gamma r_{t + 2} + \gamma^2 r_{t + 3} + \cdots\\
        &= \sum_{k=0}^{\infty}\gamma^k r_{t + k + 1}
\end{aligned}$$

- $V$ is the state value function, return the expected value from starting in state $s$ (The long-term value of state $s$):
$$\begin{aligned}
    V(s) &= \mathbb{E}[G_t | s_t = s] \\
    &= \mathbb{E}[r_{t + 1} + \gamma r_{t + 2} + \gamma^2 r_{t + 3} + \cdots | S_t = s]\\
    &= \mathbb{E}[r_{t + 1} + \gamma G_{t+1}| S_t = s]\\
    &= \mathbb{E}[r_{t + 1} + \gamma V(S_{t+1})| S_t = s]
\end{aligned}$$

- Bellman equation:
$$\begin{aligned}V(s) &= \underbrace{R(s)}_{\text{Immediate reward}} + \underbrace{\gamma \sum_{s^{\prime}\in S} P(s^{\prime} | s) V(s^{\prime})}_{\text{Discounted sum of future rewards}}\\
  \begin{bmatrix}V(s_1) \\V(s_2) \\\vdots\\V(s_N)\end{bmatrix}&= 
  \begin{bmatrix}R(s_1) \\R(s_2) \\\vdots\\R(s_N)\end{bmatrix} + \gamma\begin{bmatrix}
        p(s_1 | s_1) & p(s_2 | s_1) & \cdots & p(s_N | s_1)\\
        p(s_1 | s_2) & p(s_2 | s_2) & \cdots & p(s_N | s_2)\\
        \vdots       & \vdots       & \ddots & \vdots\\
        p(s_1 | s_N) & p(s_2 | s_N) & \cdots & p(s_N | s_N)
        \end{bmatrix}

        \begin{bmatrix}V(s_1) \\V(s_2) \\\vdots\\V(s_N)\end{bmatrix}\\

        V &= R + \gamma PV\\
        V & = (I - \gamma P)^{-1} R
  
  \end{aligned}$$
-  $q$ is the action value function.
$$\begin{aligned}
q(s, a) &= \mathbb{E} [G_t | S_t = s, A_t = a]\\
&= \mathbb{E}[r_{t + 1} + \gamma r_{t + 2} + \gamma^2 r_{t + 3} + \cdots | S_t = s, A_t = a]\\
&= \mathbb{E}[r_{t + 1} + \gamma q(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
\end{aligned}
$$

- Policies fully define the behaviour of an agent. Policies are how an agent chooses actions decides how to behave in each state.
$$
 \pi(a|s) = P(A_t = a| S_t = s) 
$$

- Bellman equation:
  - $V_{\pi}(s)$: The expected discounted reward earned from the environment, if the agent **starts in state s and makes decisions according to policy $\pi$ thereafter**.
  - $q_{\pi}(s, a)$: The expected discounted reward earned from the
environment, if the agent **takes action a from state s and makes decisions according to policy π thereafter**.

$$
\begin{aligned}
V_{\pi}(s) &= \sum_{a^{\prime} \in A} \pi(a^{\prime} | s) \cdot \sum_{s^{\prime} \in S} p(s^{\prime}, r | s, a^\prime) \cdot [r + \gamma V_{\pi}(s^\prime)]\\
& = \sum_{a^{\prime} \in A} \pi(a^{\prime} | s) \cdot q_{\pi}(s, a)\\

q_{\pi}(s, a) &= \sum_{s^{\prime} \in S} p(s^{\prime} | s, a) \cdot [r_{s, a} + \gamma \sum \pi(a^{\prime}| s^{\prime}) \cdot q_{\pi} (s^{\prime}, a^{\prime})]\\
& = \sum_{s^{\prime} \in S} p(s^{\prime} | s, a)\cdot [r + \gamma V_{\pi}(s^{\prime})]
\end{aligned}
$$

$$
\begin{aligned}
V_{\pi}(s)
 & = 
\sum_{a^{\prime} \in A}  P(a^{\prime} | s)
\begin{bmatrix}
p(s_1 | s_1, a^{\prime}) & p(s_2 | s_1, a^{\prime}) & \cdots & p(s_N | s_1, a^{\prime})
\end{bmatrix}

(
\begin{bmatrix}
R(s_1, a^{\prime})\\
R(s_2, a^{\prime})\\
\vdots\\
R(s_N, a^{\prime})\\
\end{bmatrix} + \gamma 
\begin{bmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_N) 
\end{bmatrix}
)

\end{aligned}
$$

- Example:
<div align=center>
        <img src ="32.png" width="400" height ="250"/>
</div>

$$
\begin{aligned}
V_{\pi}(s) = &0.5 * 0.1 * (1 + \gamma * 5.1) + 0.5 * 0.9 * (-2 + \gamma * -2.8)\\
 &+ 0.5 * 0.3 * (5 - \gamma * 0.3) + \cdots
\end{aligned}

$$
<div align=center>
        <img src ="33.png" width="400" height ="250"/>
</div>

$$
\begin{aligned}
q_{\pi}(s, a) = &(0.5 \cdot [3 + \gamma (0.933 * 7.7 + 0.033 * 1.2 + 0.033 * 2.1)])\\
& + 0.2 \cdot [-1 + \gamma(0.05 * 0.2 + 0.95 * 0.5)]\\
& + 0.3 \cdot [1.5 + \gamma(0.95 * 3.4 + 0.05 * -9.2)]
\end{aligned}
$$