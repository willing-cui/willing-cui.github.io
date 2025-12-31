## Write before Writing

It has been a long time since I first heard about, became interested in, and began learning Reinforcement Learning (RL) on my own—likely during my second or third year of undergraduate studies. After starting my postgraduate studies, I took RL courses taught by Professor Bolei Zhou, which helped me systematically understand the core concepts of RL.

Currently, large language models (LLMs) are highly popular in both industrial and academic fields, where RL is being deployed to enhance model training processes.

In any case, this article serves as a review of some fundamental RL concepts for me. As the Chinese saying goes, "好记性不如烂笔头" ("A sharp memory is no match for a dull pen"). With this in mind, I am restarting my self-learning journey.

## Multi-armed Bandit Problem

The Multi-armed Bandit problem is a basic and classical problem in RL. It is described as a **stateless setting** of RL. The stateless means, the problem does not involve state transitioning from one to another, unlike more complex RL problems.

<span class="image main">
<img class="main img-in-blog" style="max-width: 50%" src="./blogs/RL_Multi-Armed_Bandit/Las_Vegas_slot_machines.jpg" alt="Rewards" />
<i>By <a href="https://en.wikipedia.org/wiki/User:Yamaguchi%E5%85%88%E7%94%9F" class="extiw" title="en:User:Yamaguchi先生">Yamaguchi先生</a>, <a href="http://creativecommons.org/licenses/by-sa/3.0/" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=57295504">Link</a></i>
</span>

Here's an overview of the problem:

**Setup**:

* An agent is faced with multiple options, often referred to as "arms." Each arm corresponds to an action the agent can take.
* Pulling an arm results in a reward that is generated according to an unknown probability distribution specific to that arm.

**Objective**:

* The agent must **maximize cumulative rewards** over a finite number of interactions, $T$, by deciding which arm to pull at each step.

**Challenges**:

* The agent does not initially know the reward distributions of the arms, so it must balance **exploration** (trying different arms to gather information) with **exploitation** (choosing the arm that appears to provide the best reward based on current knowledge).

**Strategies**:   Various strategies are used to address the exploration-exploitation trade-off, such as:

* **$\epsilon$-greedy**: The agent selects the best-known arm most of the time but occasionally explores randomly.
* **Thompson sampling**: The agent uses probability distributions to evaluate and select arms based on their likelihood of being optimal.

### Thompson Sampling

Thompson Sampling is an elegant and effective algorithm for solving the exploration-exploitation dilemma in reinforcement learning and decision-making problems like the multi-armed bandit (MAB).

**Probabilistic Beliefs**:

* Each arm in the MAB problem has an associated probability distribution that represents the agent's belief about the likelihood of rewards from pulling that arm.
* Initially, these beliefs are broad and uncertain, typically modeled using prior distributions like the Beta distribution (common for Bernoulli rewards).

**Sampling**:

* At each time step $t$, the agent samples a value for each arm from its respective probability distribution.
* These samples represent the agent's current estimate of the potential reward for each arm.

**Action Selection**:

* The agent selects the arm with the highest sampled value, effectively treating it as the best option at that moment.

**Updating Beliefs**:

* After pulling an arm, the agent observes the reward and updates its belief distribution for that arm based on the new data.
* For example, if Beta distributions are used, this update process is straightforward and computationally efficient.

**Why It Works**

Thompson Sampling balances exploration and exploitation inherently:

* The sampling process encourages trying arms with high uncertainty since they have the potential for higher rewards. This is exploration.
* The probabilistic nature also favors arms with higher average rewards, enabling exploitation of known good arms.

### Mathematical Expression 

**Bayesian Approach & Posterior Updates**  

TS adopts a **Bayesian framework**:  

* **Prior belief** over $\theta_i$: $P(\theta_i)$.  

* **Posterior belief** after observing data $\mathcal{D}_t$:  
  $P(\theta_i | \mathcal{D}_t) \propto P(\mathcal{D}_t | \theta_i) P(\theta_i)$,
  where $\mathcal{D}_t = \{(a_s, r_s)\}^t\_{s=1}$ is the history of actions and rewards.  

For **Bernoulli bandits** (binary rewards), if rewards are modeled as $r_i \sim \text{Bernoulli}(\theta_i)$, we often use a **Beta prior**:  
$
\theta_i \sim \text{Beta}(\alpha_i, \beta_i)
$  
After observing $S_i$ successes and $F_i$ failures from arm $i$, the posterior is:  
$
\theta_i | \mathcal{D}_t \sim \text{Beta}(\alpha_i + S_i, \beta_i + F_i)
$  

**Thompson Sampling Algorithm**  

At each step $t$:  
1. **Sample** a parameter $\hat{\theta}_i \sim P(\theta_i | \mathcal{D}_t)$ for each arm $i$.  
2. **Select** the arm with the highest sampled value:  
   $
   a_t = \arg\max_{i} \hat{\theta}_i
   $  
3. **Observe** reward $r_t$ and update the posterior:  
   $P(\theta\_{a_t} | \mathcal{D}\_{t+1}) \propto P(r_t|\theta\_{a_t}) P(\theta\_{a_t} | \mathcal{D}_t)$

### Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

class BernoulliBanditEnv:
    def __init__(self, K):
        self.K = K
        self.probs = np.random.uniform(0, 1, K)  # True success probabilities for each arm

    def generate_reward(self, arm):
        # Returns 1 with probability p, 0 otherwise
        return np.random.binomial(1, self.probs[arm])

class EpsilonGreedyAgent:
    def __init__(self, K, epsilon=0.1):
        self.K = K
        self.epsilon = epsilon
        self.counts = np.zeros(K)  # Number of times each arm was pulled
        self.values = np.zeros(K)  # Empirical mean reward for each arm

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        # Update the empirical mean using incremental update formula
        self.values[arm] = ((n - 1) * self.values[arm] + reward) / n

    def act(self):
        if np.random.random() < self.epsilon:
            # Exploration: choose random arm
            return np.random.randint(self.K)
        else:
            # Exploitation: choose arm with the highest empirical mean
            return np.argmax(self.values)

class ThompsonSamplingAgent:
    def __init__(self, K):
        self.K = K
        # Beta parameters: alpha = successes + 1, beta = failures + 1
        self.alphas = np.ones(K)
        self.betas = np.ones(K)

    def update(self, arm, reward):
        if reward == 1:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1

    def act(self):
        # Sample from each arm's posterior beta distribution
        samples = [np.random.beta(self.alphas[a], self.betas[a]) for a in range(self.K)]
        return np.argmax(samples)

def run_experiment(env, agent, T):
    rewards = np.zeros(T)
    for t in range(T):
        arm = agent.act()
        reward = env.generate_reward(arm)
        agent.update(arm, reward)
        rewards[t] = reward
    return rewards

def run_multiple_experiments(K_values, T, num_experiments=10):
    results = {}

    for K in K_values:
        print(f"Running experiments for K={K}...")
        eps_rewards = np.zeros((num_experiments, T))
        ts_rewards = np.zeros((num_experiments, T))
        opt_rewards = np.zeros((num_experiments, T))

        for exp in range(num_experiments):
            # Same environment for both agents in each experiment
            env = BernoulliBanditEnv(K)

            # Epsilon-greedy
            eps_agent = EpsilonGreedyAgent(K, epsilon=0.1)
            eps_rewards[exp] = run_experiment(env, eps_agent, T)

            # Thompson sampling
            ts_agent = ThompsonSamplingAgent(K)
            ts_rewards[exp] = run_experiment(env, ts_agent, T)

            # Optimal rewards
            opt_rewards[exp] = np.max(env.probs) * np.ones(T)

        # Average across experiments
        avg_eps = np.mean(eps_rewards, axis=0)
        avg_ts = np.mean(ts_rewards, axis=0)
        avg_opt = np.mean(opt_rewards, axis=0)

        results[K] = {
            'epsilon_greedy': avg_eps,
            'thompson_sampling': avg_ts,
            'optimal': avg_opt
        }

    return results

def plot_results(results, K_values):
    plt.figure(1, figsize=(8, 10))

    for i, K in enumerate(K_values):
        plt.subplot(len(K_values), 1, i + 1)
        data = results[K]

        # Cumulative rewards
        cum_eps = np.cumsum(data['epsilon_greedy'])
        cum_ts = np.cumsum(data['thompson_sampling'])
        cum_optimal = np.cumsum(data['optimal'])

        plt.plot(cum_eps, label=f'ε-greedy (ε=0.1)')
        plt.plot(cum_ts, label='Thompson Sampling')
        plt.plot(cum_optimal, label='Optimal', linestyle='--')

        plt.title(f'K={K} Arms')
        plt.xlabel('Time step')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    plt.figure(2, figsize=(8, 10))

    for i, K in enumerate(K_values):
        plt.subplot(len(K_values), 1, i + 1)
        data = results[K]

        # Rewards
        rwd_eps = data['epsilon_greedy']
        rwd_ts = data['thompson_sampling']
        rwd_optimal = data['optimal']

        plt.plot(rwd_eps, label=f'ε-greedy (ε=0.1)', alpha = 0.75)
        plt.plot(rwd_ts, label='Thompson Sampling', alpha = 0.75)
        plt.plot(rwd_optimal, label='Optimal', linestyle='--')

        plt.title(f'K={K} Arms')
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# Code Test
if __name__ == "__main__":
    # Parameters
    K_values = [5, 10, 20]
    T = 5000
    num_experiments = 100  # Number of experiments to average over

    # Run experiments
    results = run_multiple_experiments(K_values, T, num_experiments)

    # Plot results
    plot_results(results, K_values)
```

#### Simulation Results

The reward obtained by the agent varies across different simulation environment settings and strategies.

<span class="image main">
<img class="main img-in-blog" style="max-width: 45rem" src="./blogs/RL_Multi-Armed_Bandit/Simulation_Results_2.webp" alt="Rewards" />
</span>

The accumulated rewards.

<span class="image main">
<img class="main img-in-blog" style="max-width: 45rem" src="./blogs/RL_Multi-Armed_Bandit/Simulation_Results_1.webp" alt="Accumulated rewards" />
</span>

### Analysis

**Performance Comparison:**

1. Thompson Sampling consistently outperforms $\epsilon$-greedy across all values of $K$.
2. The performance gap between the two strategies widens as $K$ increases.
3. Both approaches eventually converge toward the optimal strategy, but Thompson sampling does so faster.

**Why Thompson Sampling Performs Better:**

1. It maintains a probability distribution over each arm's true reward rate.
2. Exploration is naturally guided by uncertainty - arms with higher uncertainty get explored more.
3. Automatically balances exploration and exploitation without a fixed parameter like $\epsilon$, making it more efficient in larger action spaces (higher $K$ values).

**$\epsilon$-greedy Limitations:**

1. Fixed exploration rate ($\epsilon=0.1$) means it continues random exploration even after identifying good arms.
2. Doesn't account for uncertainty in estimates - treats all estimates as equally reliable, which wastes pulls on suboptimal arms.

## Basic Concepts

### Conjugate Prior

In Bayesian probability theory, if given a likelihood function:  $p(x|\theta)$, the posterior distribution $p(\theta|x)$ is in the same probability distribution family as the prior probability distribution $p(\theta)$.

* The prior and posterior are then called conjugate distributions with respect to that likelihood function.

* And the prior is called a conjugate prior for the likelihood function.

<span class="image main medium">
<img class="main img-in-blog medium" style="max-width: 30rem" src="./blogs/RL_Multi-Armed_Bandit/Prior_Likelihood_Posterior.webp" alt="The relationship between the Prior, Likelihood and Posterior probability." />
</span>

### Beta Distribution

The Beta distribution is a versatile probability distribution that is defined on the interval $[0, 1]$ and is widely used in Bayesian statistics and machine learning. It is particularly valuable for modeling probabilities and proportions.

#### Definition

The probability density function (PDF) of the Beta distribution is given by:

<div class="formula-in-blog">

$$
f(x; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha - 1}(1 - x)^{\beta - 1}, \quad \text{for } x \in [0, 1], 
$$

</div>

where:

* $\alpha > 0$ and $\beta > 0$ are shape parameters.
* $\Gamma(\cdot)$ is the Gamma function, which generalizes the factorial to non-integer values.

#### Key Features

* **Range**: The Beta distribution is defined only within $[0, 1]$, making it suitable for modeling probabilities or proportions.
* **Shape**: The shape of the distribution depends on $\alpha$ and $\beta$:
  + If $\alpha = \beta$, the distribution is symmetric.
  + If $\alpha > \beta$, the distribution is skewed to the right.
  + If $\alpha < \beta$, the distribution is skewed to the left.
  + Larger values of $\alpha$ and $\beta$ lead to a more concentrated distribution.

#### Mean and Variance

For the Beta distribution:

* Mean: $\frac{\alpha}{\alpha + \beta}$
* Variance: $\frac{\alpha \beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}$

#### Applications

The Beta distribution has practical applications in various areas:

1. **Bayesian Inference**: Used as a prior distribution for binary outcomes (e.g., success/failure scenarios) in **conjunction with the Bernoulli distribution**.
2. **Thompson Sampling**: In multi-armed bandit problems, Beta distributions are updated to model the success probabilities of arms.
3. **Proportion Modeling**: Suitable for modeling proportions like click-through rates, probabilities, and ratios.

### Bayes' rule

Bayes' Rule, also known as Bayes' Theorem, is a fundamental concept in probability theory and statistics. It provides a way to update the probability of a hypothesis based on new evidence. It’s a cornerstone of Bayesian statistics and is widely used in various fields, from machine learning to medicine. Here's the formula:

<div class="formula-in-blog">

$$
P(H|E)= \frac{P(E|H) \cdot P(H)}{P(E)}
$$

</div>

#### Explanation of Terms:

* $P(H|E)$: The posterior probability of hypothesis $H$, given the evidence $E$. This represents the updated belief about $H$ after considering the evidence.
* $P(E|H)$: The likelihood of observing the evidence $E$ if the hypothesis $H$ is true.
* $P(H)$: The prior probability of $H$, representing the initial belief about the hypothesis before considering any evidence.
* $P(E)$: The marginal probability of the evidence $E$, representing the total probability of $E$ under all possible hypotheses.

### Gamma Function

The Gamma function, denoted as $ \Gamma(z) $, is a key concept in mathematics, particularly in calculus, probability, and complex analysis. It serves as a continuous extension of the factorial function to non-integer values.

#### Definition

The Gamma function is defined as:

$
\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} \, dt, \quad \text{for } z > 0.
$

Here:
* $z$ is a complex or real number (with $\text{Re}(z) > 0$).
* The integral converges for these values.

#### Connection to the Factorial

For positive integers, the Gamma function relates to the factorial as follows:

$
\Gamma(n) = (n-1)!, 
$

where $n$ is a positive integer. For example:
$
\Gamma(5) = 4! = 24.
$

This relationship allows the Gamma function to generalize the factorial for non-integer and even complex inputs.

#### Properties

1. **Recurrence Relation**:  
   $\Gamma(z+1) = z\Gamma(z)$, similar to the recurrence in factorials.

2. **Special Value**:  
   $\Gamma(1) = 1$.  

3. **Reflection Formula**:  
   $\Gamma(z)\Gamma(1-z) = \frac{\pi}{\sin(\pi z)}$, connecting values at $z$ and $1-z$.
