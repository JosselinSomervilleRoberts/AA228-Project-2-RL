import numpy as np
from tqdm import tqdm


# Value iteration algorithm
def value_iteration(data_dict, num_states, gamma=0.95, max_num_iter=10000, delta_stop=0.01):
    V = np.zeros(num_states)
    for _ in tqdm(range(max_num_iter)):
        delta = 0
        old_V = V.copy()
        for idx_s in range(num_states):
            s = idx_s + 1
            Q = np.zeros(4)
            for a in data_dict[s]:
                n_total = 0
                for (sp, r) in data_dict[s][a].keys():
                    n = data_dict[s][a][(sp, r)]
                    n_total += n
                    Q[a - 1] += n * (r + gamma * old_V[sp - 1])
                Q[a - 1] /= n_total
            V[idx_s] = np.max(Q)
            delta = max(delta, abs(old_V[idx_s] - V[idx_s]))
        if delta < delta_stop:
            break
    return V

def get_policy_from_V(data_dict, V, gamma=0.95):
    policy = np.zeros(len(V), dtype=int)
    for idx_s in range(len(V)):
        s = idx_s + 1
        Q = np.zeros(4)
        for a in data_dict[s]:
            n_total = 0
            for (sp, r) in data_dict[s][a].keys():
                n = data_dict[s][a][(sp, r)]
                n_total += n
                Q[a - 1] = n * (r + gamma * V[sp - 1])
            Q[a - 1] /= n_total
        policy[idx_s] = np.argmax(Q) + 1
    return policy



if __name__ == "__main__":
    from utils import get_data_dict, save_policy
    size = "small"
    data_dict = get_data_dict(size)
    V = value_iteration(data_dict, 100, gamma=0.95, max_num_iter=1000, delta_stop=1e-10)
    print(V)
    policy = get_policy_from_V(data_dict, V, gamma=0.95)
    save_policy(policy, size)
    print(policy)