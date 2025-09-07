---
layout: default
title: Policy evaluation in Reinforcement Learning - understanding eligibility traces
tags: 
    - post
---

In Reinforcement Learning, what seems to be a powerful tool, the eligibility trace TD(λ), as described in [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) in Chapter 12.2, is a way of mixing TD(0) bootstraping with Monte-Carlo methods. It seems to be providing good flexibility and several improvements in theory. Yet, in most state of the art RL algorithms, such as [PPO](https://arxiv.org/abs/1707.06347), it is not used. In this article, we will learn what is the eligibility trace TD(λ), how it fits in the Reinforcement Learning framework, and try to answer the question: Why isn’t this method used in state of the arts algorithms?

# TL;DR
TD(λ) is a foundational algorithm in RL and remains important for understanding value function estimation. However, in modern RL, the focus has shifted toward policy optimization methods like PPO, not value function estimation, which use simpler and more effective techniques like GAE for advantage estimation. This shift is driven by practical considerations such as computational efficiency, ease of implementation, and empirical performance. TD(λ) adds complexity with an additional hyperparameter, λ, and additional complexity in maintaining the eligibility traces.

# Policy Evaluation
We can break down most Reinforcement Learning algorithms in to parts: policy evaluation and policy improvement. Most basic RL algorithms such as Q-learning, mostly rely on policy evaluation, since policy improvement is basically taking the max of the Q-values. This is is described with an iterative rule,

$$ \pi_{n+1} = \arg \max_a Q^{\pi_n}(s,a)$$

As a reminder, the goal of policy evaluation is to approximate as best as possible the true value of the state values or state-action values under the current policy. This approximation allows us the ability to quantify the error between our current estimate and the true value. Using state value functions, for a single state, this error can be written as follow:

$$ VE = v_\pi(s) - \hat{v}(s),$$

with $$v_\pi(s)$$ the true value under the policy $$\pi$$, and $$\hat{v}(s)$$ the current estimate.

Out of the above equation, the following question arise: How do we know the true value? Well, we don’t, so we have to estimate it. The way we estimate this true value will impact the efficiency of our policy evaluation algorithm. But once we an estimate of this error, many choices arise, for instance, updating our model parameters in the opposite direction of the gradient of this error (or some better form of it) to eventually reduce the error. Or, we could define our estimation as a weighted average of each successive error, in a iterative scheme. This is out of the scope for this article, but we understand that having access to this error in some shape or form is the cornerstone of many RL algorithm, and the more this error is close to its real value, the better.

# Choices for computing the true value of V(s)
Now let’s enumerate some common choices we have to compute the true value of V(s) under our current policy. Let’s recall the definition of V(s) first:

$$v_\pi(s) = \mathbb{E}[G_t | S_t = s]$$

Where $$G_t$$ are the discounted returns.

We are now ready to describe our first option,

# Monte-Carlo method

$$v_\pi(s_t) = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots + \gamma^{T-t-1}R_{T-1}$$

where $$R_t$$ is the reward received at time t, and $$\gamma$$ the discount factor.

This uses the full episode length T and should, in theory, be unbiased, over the long run, because it is the exact definition given above, and therefore, after an infinite number of episodes, should converge to the true value. However, this has major drawbacks:

* We need to wait for a full episode to end before evaluating our policy
* The convergence is prohibitively slow in practice, as for every timestep, the update is given based on one single trajectory (the following actions taken by the current policy) and takes forever to converge to the true value

# TD(0)
Another (arguably as extreme) option, is to take the TD(0) estimate:

$$v_\pi(s_t) = R_t + \gamma \hat{v}_\pi(s_{t+1})$$ 

I find it to be the most widely used approximation method. Intuitively, this take an extra bit of information (namely the reward received directly after the current state) and discounting the model’s estimation of the next state. Therefore, this is slightly closer to the true value of V(s) than our model’s current estimate. The main advantage with this is that you don’t need to wait the full episode before updating, and it extends to non episodic cases. In practice, this works better than MC methods, but this also has major drawbacks:

* This takes a lot of updates to converge to the true value because each estimate is biased
* It takes a long time for a reward received at a time T to propagate and reinforce actions taken many steps earlier. This is because for each update, only one state see its value updated.

# n-step TD
A natural question arises: instead of either taking the full T discounted rewards, or just the first one, to estimate the true value, could we take any number in between? The answer is yes, and as described in [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), this method very often improves results in practice. The form can be described with:

$$\text{n-step TD}(v_\pi(s_t)) = R_t + \gamma R_{t+1} + \cdots + \gamma^n \hat{v}(s_{t+n+1})$$

However, we are now stuck with n being fixed. Wouldn’t it be nice to blend these methods altogether and be able to weight some of them more than other to fine-tune our estimate of the true value? Sure thing, we can use the eligibility trace

# Eligibility trace as λ-returns
Intuitively, this method combines many n-step TD estimates together, and weight them, with normalization so the weights sum up to 1. It can be thought of holding some sort of decaying memory of the past experiences so after each update, more than 1 action is being reinforced from the reward. They offer many computational advantages. They are defined below,

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}$$

Looking at this equation, with λ = 0, this becomes the TD(0), and with λ = 1, we are back to MC method. The second part becomes more clear once we separate the terminal state from the sum:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1}G_{t:t+n} + \lambda^{T-t-1} G_t$$

Still, there are issues with this method. This is “forward” looking, in a sense that for each state encountered in our history, we still need to look “n” steps ahead. It also requires to wait for the end of the episode before updating. The computations of this method are similar to MC methods as everything is done at the end of the episode. Lastly, it can’t be applied to continuous problems. So, it seems we did one step forward, two back. One straightforward way would be to truncate it, because each step gets decayed anyways, so it would be reasonable. However, another solution fixes all that, the TD(λ).

# TD(λ)
In this method, we still use eligibility traces, as defined above, but we use them as a decaying additional update to our model’s parameters. Intuitively, we keep an eligibility trace vector starting being equal to our TD update, which is decaying over time, therefore providing us with a sliding window of our update vector, keeping in memory decaying TD errors. Mathematically, the eligibility trace vector is written:

$$z_0 = 0, z_t = \gamma \lambda z_{t-1} + f(S_t)$$

with $$f(S_t)$$ representing the immediate eligibility contribution of the current state.

We then use the standard TD error

$$\delta = R_{t+1} + \gamma \hat{v}(S_{t+1} - \hat{v}(S_t))$$

And update our model as such (in the case of approximation methods, where the functional f would be the gradient w.r.t model parameters):

$$w_{t+1} = w_t + \alpha \delta_t z_t$$ 

In the case of tabular methods, f would be a one-hot vector with 1 on the current s, and what would be updated are the current state values of the state, not the model weights.

From those equations we can see that if λ=1 then we are in the case of MC method, but one which does not need to wait for the full episode and can update at every timestep. On the opposite, if λ=0, we are back in the standard TD(0). We can further improve or tweak this method, and more advanced algorithms are described in [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), Chapter 12.3–12.8.

# Why aren’t eligibility traces used in modern state of the art algorithms?
First of all let’s note that eligibility traces are only useful in value-based method, i.e algorithms which focus on updating the estimates of state or state-action values. A lot of modern algorithms, such as PPO, are policy-based method, which focuses on directly improving the policy. However, PPO uses an actor-critic framework, from which the critic’s network is required to provide information about the advantage function. This can be thought as a similar thing to prediction objective VE. But, eligibility traces are quite expensive to keep track of, adding an additional hyperparameter, and reinforcement learning is already complex enough so that modern algorithms tend to prefer simplicity in most case. On top of that, PPO uses GAE for advantage estimation, which is supposedly more effective in practice, and was arguably influenced by eligibility traces.

It is still important to understand how eligibility traces work, because they provide a good insight into how complex Reinforcement learning can become. They are more than just a mechanism for credit assignment, they also highlight a very important question: To what extent should past actions be remembered and reinforced when rewards are received? This question is central in balancing short-term and long-term dependencies, potentially important for planning, and can inspire future advancements in RL.