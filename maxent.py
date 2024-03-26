import numpy as np

def feat_exp(trajs, feats):
    num_feats = feats.shape[1]
    fe = np.zeros(num_feats)
    for t in trajs:
        for s in t.states():
            fe += feats[s, :]
    return fe / len(trajs)

def init_probs(trajs, num_states):
    p = np.zeros(num_states)
    for t in trajs:
        p[t.transitions()[0][0]] += 1.0
    return p / len(trajs)

def svf_from_policy(trans_probs, init_probs, terminals, action_probs, eps=1e-5):
    num_states, _, num_actions = trans_probs.shape
    trans_probs[terminals, :, :] = 0.0
    p_trans = [np.array(trans_probs[:, :, a]) for a in range(num_actions)]
    d = np.zeros(num_states)
    delta = np.inf
    while delta > eps:
        d_ = [p_trans[a].T.dot(action_probs[:, a] * d) for a in range(num_actions)]
        d_ = init_probs + np.array(d_).sum(axis=0)
        delta, d = np.max(np.abs(d_ - d)), d_
    return d

def action_probs(trans_probs, terminals, rewards):
    num_states, _, num_actions = trans_probs.shape
    er = np.exp(rewards)
    p = [np.array(trans_probs[:, :, a]) for a in range(num_actions)]
    zs = np.zeros(num_states)
    zs[terminals] = 1.0
    for _ in range(2 * num_states):
        za = np.array([er * p[a].dot(zs) for a in range(num_actions)]).T
        zs = za.sum(axis=1)
    return za / zs[:, None]

def expected_svf(trans_probs, init_probs, terminals, rewards, eps=1e-5):
    action_ps = action_probs(trans_probs, terminals, rewards)
    return svf_from_policy(trans_probs, init_probs, terminals, action_ps, eps)

def irl_main(trans_probs, feats, terminals, trajs, optim, init, eps=1e-4, eps_svf=1e-5):
    num_states, _, num_actions = trans_probs.shape
    num_feats = feats.shape[1]
    e_feats = feat_exp(trajs, feats)
    init_ps = init_probs(trajs, num_states)
    theta = init(num_feats)
    delta = np.inf
    optim.reset(theta)
    while delta > eps:
        theta_old = theta.copy()
        rewards = feats.dot(theta)
        grad = e_feats - feats.T.dot(expected_svf(trans_probs, init_ps, terminals, rewards, eps_svf))
        optim.step(grad)
        delta = np.max(np.abs(theta_old - theta))
    return feats.dot(theta)

def softmax(x1, x2):
    x_max = np.maximum(x1, x2)
    return x_max + np.log(1.0 + np.exp(np.minimum(x1, x2) - x_max))

def causal_action_probs(trans_probs, terminals, rewards, discount, eps=1e-5):
    num_states, _, num_actions = trans_probs.shape
    reward_terminal = -np.inf * np.ones(num_states)
    reward_terminal[terminals] = 0.0
    p = [np.array(trans_probs[:, :, a]) for a in range(num_actions)]
    v = -1e200 * np.ones(num_states)
    delta = np.inf
    while delta > eps:
        v_old = v
        q = np.array([rewards + discount * p[a].dot(v_old) for a in range(num_actions)]).T
        v = reward_terminal
        for a in range(num_actions):
            v = softmax(v, q[:, a])
        v = np.array(v, dtype=float)
        delta = np.max(np.abs(v - v_old))
    return np.exp(q - v[:, None])

def expected_causal_svf(trans_probs, init_probs, terminals, rewards, discount, eps_lap=1e-5, eps_svf=1e-5):
    action_ps = causal_action_probs(trans_probs, terminals, rewards, discount, eps_lap)
    return svf_from_policy(trans_probs, init_probs, terminals, action_ps, eps_svf)

def irl_causal(trans_probs, feats, terminals, trajs, optim, init, discount, eps=1e-4, eps_svf=1e-5, eps_lap=1e-5):
    num_states, _, num_actions = trans_probs.shape
    num_feats = feats.shape[1]
    e_feats = feat_exp(trajs, feats)
    init_ps = init_probs(trajs, num_states)
    theta = init(num_feats)
    delta = np.inf
    optim.reset(theta)
    while delta > eps:
        theta_old = theta.copy()
        rewards = feats.dot(theta)
        grad = e_feats - feats.T.dot(expected_causal_svf(trans_probs, init_ps, terminals, rewards, discount, eps_lap, eps_svf))
        optim.step(grad)
        delta = np.max(np.abs(theta_old - theta))
    return feats.dot(theta)

