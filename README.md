### Soft Actor Critic
------------
This is a pytorch implementation of [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf) running on the walked environment in Unity ML. <br>
This doc includes SAC module usage and instructions to run the experiments.
##### TO DOs
- [ ] add requirements and dependencies <br>
- [ ] add curves and model weights <br>
- [ ] add documentations <br>

#### SAC Usage
```
agent = SAC(state_dim, action_dim, args)
for _ in t:
    actions = agent.act(states)
    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.step(states, actions, rewards, next_states, dones)
while evaluate:
    actions = agent.act(eval = True)
```

#### Usage
```
cd sac
python walker.py --system <linux/window>
```
