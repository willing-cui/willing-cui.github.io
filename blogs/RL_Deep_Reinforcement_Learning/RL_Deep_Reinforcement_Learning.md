The **target network** is a crucial component in Deep Q-Networks (DQN) that helps stabilize training by **reducing harmful correlations** and **preventing feedback loops** that can occur when using a single network for both action selection and value estimation.

## How the Target Network Works

1. **Two Networks**:
   - **Policy Network (Online Network)**: Continuously updated during training.
   - **Target Network**: A delayed copy of the policy network, updated less frequently.
2. **Temporal Difference (TD) Learning**:
   - In standard Q-learning, the target Q-value is computed as:Target=*r*+*γ*⋅*a*′max*Q*(*s*′,*a*′)
   - If we use the **same network** to compute both current and next-state Q-values, the target keeps shifting, leading to instability.
3. **Fixed Targets**:
   - The target network provides **stable Q-value estimates** for bootstrapping.
   - The target Q-value is computed as:Target=*r*+*γ*⋅*a*′max*Q*target(*s*′,*a*′)
   - Since the target network updates **less frequently**, the learning process is smoother.

## Why It Stabilizes Training



1. **Reduces Moving Target Problem**:
   - Without a target network, the Q-network is chasing its own updates, leading to oscillations or divergence.
   - The target network acts like a "snapshot" of the policy network, providing stable targets.
2. **Breaks Correlations**:
   - In standard Q-learning, consecutive states are highly correlated (e.g., frames in a game).
   - The target network helps **decouple** the estimation of future rewards from immediate updates.
3. **Prevents Feedback Loops**:
   - If the policy network overestimates Q-values, it can lead to a feedback loop where bad estimates reinforce themselves.
   - The target network mitigates this by providing **less frequently updated** (and thus more conservative) estimates.