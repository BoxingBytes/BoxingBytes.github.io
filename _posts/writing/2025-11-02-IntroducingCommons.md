---
layout: post
title: Building the Harvest Commons — a minimal MARL society in C
tags:
    - blog
    - writing
    - highlight
---

# Goal for this article

The goal of this post is to introduce the first version of **Harvest Commons**, a multi-agent environment built in C and interfaced through [pufferlib](https://puffer.ai/). This environment is designed to explore how agents survive, compete, and cooperate when facing a shared renewable resource — a simplified simulation of a **common-pool resource economy**. I already did implement & wrote about a barebone version of this. This time, I intend to gradually evolve the environment. 

We’ll start small: four agents, one biome, one regrowing resource, one rule — don’t die.

# The Core Problem 

Imagine four agents dropped into a world with limited, regenerating food. They need energy to survive, but the food grows slowly. Do they:

* Coordinate to harvest sustainably?
* Compete and deplete the resource?
* Free-ride while others do the work?

This mirrors real-world dilemmas: fisheries, forests, public infrastructure.

# The idea

The Harvest Commons is a minimal world to study **emergent cooperation** and **resource sustainability**.  
Each agent loses 1 HP per step, and dies at 0 HP.  
To survive, agents must **collect food**, store it, and **eat** when needed.  
Food regrows over time in its biome, but can be depleted if agents overharvest.

This setup creates a simple tension: should an agent act greedily now, or conserve resources for later?  
The environment is intentionally minimal to isolate survival dynamics before adding social mechanisms.

# Mechanics

| Mechanic | Description |
|-----------|--------------|
| **Grid** | Discrete 2D world |
| **Agents** | 4 agents with position, HP, inventory, and ID. |
| **Food** | Regrows probabilistically each step in its corresponding biome.|
| **Inventory** | Agents must store food before they can eat it. |
| **HP system** | HP decreases each turn, increases when eating (+10 HP). Max HP is 100. |
| **Stealing** | Agents can steal another agent's food from its inventory if facing it. |
| **Reward** | Very sparse: only −1 for dying (no positive reward yet). |
| **Actions** | Each agent can: Move up/down/left/right, stay idle, eat food, or interact (collect food or steal) |
| **Observations** | Agents have a vision of 3 around them. For each tile they see the type, and if it's an agent, the HP and food in its inventory. |


A minimal setup like this allows us to study *pure emergence*: can a policy network learn to stay alive and harvest sustainably with no explicit guidance?

# Rendering

Here’s what the world looks like right now:

![harvest_commons_v0](/assets/images/commons/harvestcomons_v0.png)

Food only regrows on dirt tiles (bottom right). 
Green and grey tiles aren't used for now. We can observe the health and food in inventory for each agent. 

# Training with sparse rewards

I started with the most extreme sparse reward: agents only get −1 when they die. No reward for collecting food, eating, or surviving longer. This tests if pure survival pressure is enough to learn coordination. Food currently can not be depleted, as it regrows no matter what with 10% probability each step on its biome.
Early observations:

<video width="100%" preload="auto" muted controls playsinline autoplay>
  <source src="{{ '/assets/videos/commons/v0_stable_opt.mp4' | relative_url }}" type="video/mp4">
</video>

**Agents learn to survive!** 

However, there is something to be said on the learning curves: 

![learning_curves](/assets/images/commons/harvestcommons_v0_score_curves_unstable.png)

As you can see, runs are very unstable. And this is a picture with everything seeded and the same hyperparams. What I suspect happens is the following: Due to very sparse rewards, once the agents learn to survive, the only way to improve their reward is to improve entropy (The training algo is PPO based with entropy loss). By doing so they inevitably fall off the optimal policy, from which they have to re-learn, and the cycle continue. However this hypothesis seems to be wrong: when I put side by side entropy and score, high entropy spikes do not match low scores area. 

Another interesting observation is the steal ratio: 

![steal_curves](/assets/images/commons/v0_stealing_curves.png)

Stealing is only available when one agent is facing another agent. When an agent steal, it gets the other agent's entire inventory. The y-axis is in %/100.I called that stealing, but this can also be thought as a collaborative strategy because it can help feed an agent that is starving. However, in this setup, it is purely competitive, as the reward is only -1 on death. Surprinsingly, score and steal curve seems to align. I'm not sure why, but it means the more agent steal, the more each agent seems to survive on average. 

# Why this setup?

When do agents learn to cooperate purely from environmental constraints? No hand-crafted reward for "being nice." Just survival.
This environment is the foundation for more complex mechanics:

* Energy systems (sleep, fatigue)
* Fire maintenance (collective goods)
* Multi-home dynamics (territorial behavior)
* Trading and reputation (social structures)

But first, I need to understand what incentives make a stable society possible.

# Next

In the next post, I'll share results from training with sparse rewards and explore different incentive structures:

What happens with pure survival pressure?
Do denser rewards help or hurt coordination?
Can we find a reward that leads to sustainable harvesting?

The goal is simple: find the minimal incentive structure that produces intelligent, cooperative behavior.
Stay tuned.

# Fun bonus

I found this weird behavior when looking at policies around the time the score was going down

<video width="100%" preload="auto" muted controls playsinline autoplay>
  <source src="{{ '/assets/videos/commons/v0_rush_opt.mp4' | relative_url }}" type="video/mp4">
</video>

Agents tend to wait the last minute before having no HP to rush down & get food. No wonder this strategy is more risky and more agents die! However, I have no idea why agents do such a thing.

# Influences & References

Assets used are NMMO3s ones available in pufferlib

Training is done using pufferlib main trainer in current 3.0 version
