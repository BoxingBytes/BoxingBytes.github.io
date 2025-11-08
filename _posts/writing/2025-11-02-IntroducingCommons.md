---
layout: post
title: Building a minimal multi agent society
tags:
    - blog
    - writing
    - highlight
---

I introduce the first version of a multi-agent environment built in C and built on top of [pufferlib](https://puffer.ai/). This environment is designed to explore how agents survive, compete & cooperate in a complex environment across different incentives. I already wrote about a barebone version of this [here]({% post_url 2025-03-27-buildCenv %}). This time, I intend to gradually evolve the environment. 

This can be applied to a wide range of real-world problems. In video-games, you might want to evolve a stable, rich environment where NPCS have organic behaviors. In smart-city, economy, logistics, you can simulate your system and look at what incentives produces the desire behaviour. This can be used to study the emergence of collective intelligence, and more.

# The Core Problem 

We’ll start small: four agents, one regrowing resource, one rule — don’t die. They are dropped into a world with limited resources. They need energy to survive, but the food grows slowly. Under what incentives do they manage to find a stable system, where every agent get its own share of the resource without depleting the environment? What solutions do these agents find to inherent problems of this world?   

Each agent loses 1 HP per step, and dies at 0 HP. To survive, agents must **collect food**, store it, and **eat** when needed.
Food regrows over time in its biome, but can be depleted if agents overharvest. The environment is intentionally minimal to isolate survival dynamics before adding social mechanisms.

# Mechanics

| Mechanic | Description |
|-----------|--------------|
| **Grid** | Discrete 2D world |
| **Agents** | 4 agents with position, HP, inventory, and ID. |
| **Food** | Regrows probabilistically each step in its corresponding biome.|
| **Inventory** | Agents must store food before they can eat it. |
| **HP system** | HP decreases each turn, increases when eating (+20 HP). Max HP is 100. |
| **Stealing** | Agents can steal another agent's food from its inventory if facing it. |
| **Reward** | Very sparse: only −1 for dying (no positive reward yet). |
| **Actions** | Each agent can: Move up/down/left/right, stay idle, eat food, or interact (collect food or steal) |
| **Observations** | Agents have a vision of 3 around them. For each tile they see the type, and if it's an agent, the HP and food in its inventory. |

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

Stealing is only available when one agent is facing another agent. When an agent steal, it gets the other agent's entire inventory. The y-axis is percent.I called that stealing, but this can also be thought as a collaborative strategy because it can help feed an agent that is starving. I'm not sure why, but it means the more agent steal, the more each agent seems to survive on average. 

# Next

This environment is the foundation for more complex mechanics:

* Energy systems (sleep, fatigue)
* Fire maintenance (collective goods)
* Multi-home dynamics (territorial behavior)
* Trading and reputation (social structures)

But first, I need to understand what can make those runs stable. This is what I'll explore in the next post.

# Fun bonus

I found this weird behavior when looking at policies around the time the score was going down

<video width="100%" preload="auto" muted controls playsinline autoplay>
  <source src="{{ '/assets/videos/commons/v0_rush_opt.mp4' | relative_url }}" type="video/mp4">
</video>

Agents tend to wait the last minute before having no HP to rush down & get food. No wonder this strategy is more risky and more agents die! However, I have no idea why agents do such a thing.

# Influences & References

Assets used are NMMO3s ones available in pufferlib

Training is done using pufferlib main trainer in current 3.0 version
