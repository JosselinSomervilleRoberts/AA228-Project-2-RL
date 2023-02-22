import numpy as np
from tqdm import tqdm
from utils import sample_action


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
                Q[a - 1] += n * (r + gamma * V[sp - 1])
            Q[a - 1] /= n_total
        policy[idx_s] = np.argmax(Q) + 1
    return policy


def policy_iteration(data_dict, num_states, num_actions, gamma=0.95, max_num_iter=10000, max_num_iter_policy=50, delta_stop=0.01):
    V = np.zeros(num_states)
    policy = [sample_action(data_dict, idx_s + 1) for idx_s in range(num_states)]

    for _ in range(max_num_iter_policy):
        # Policy evaluation
        V = np.zeros(num_states)
        for _ in tqdm(range(max_num_iter)):
            delta = 0
            old_V = V.copy()
            V = np.zeros(num_states)
            for idx_s in range(num_states):
                s = idx_s + 1
                a = policy[idx_s]
                n_total = 0
                for (sp, r) in data_dict[s][a].keys():
                    n = data_dict[s][a][(sp, r)]
                    n_total += n
                    V[idx_s] += n * (r + gamma * old_V[sp - 1])
                V[idx_s] /= n_total
                delta = max(delta, abs(old_V[idx_s] - V[idx_s]))
            if delta < delta_stop:
                break
        
        # Policy improvement
        stable = True
        for idx_s in range(num_states):
            s = idx_s + 1
            Q = np.zeros(num_actions)
            for a in data_dict[s]:
                n_total = 0
                for (sp, r) in data_dict[s][a].keys():
                    n = data_dict[s][a][(sp, r)]
                    n_total += n
                    Q[a - 1] += n * (r + gamma * V[sp - 1])
                Q[a - 1] /= n_total
            old_pi_s = policy[idx_s]
            policy[idx_s] = np.argmax(Q) + 1
            if old_pi_s != policy[idx_s]:
                stable = False

        if stable:
            break

    return V, policy



if __name__ == "__main__":
    from utils import get_data_dict, save_policy
    size = "small"
    data_dict = get_data_dict(size)

    # Value iteration
    print("Value iteration...")
    V1 = value_iteration(data_dict, 100, gamma=0.95, max_num_iter=1000, delta_stop=1e-10)
    policy1 = get_policy_from_V(data_dict, V1, gamma=0.95)
    save_policy(policy1, size + "_value_iter")
    print("Value iteration done")
    print("V1:", V1)

    # Policy iteration
    print("\n\nPolicy iteration...")
    V2, policy2 = policy_iteration(data_dict, 100, 4, gamma=0.95, max_num_iter=1000, max_num_iter_policy=50, delta_stop=1e-10)
    save_policy(policy1, size + "_policy_iter")
    print("Policy iteration done")
    print("V2:", V2)

    # Check if the two policies are the same
    print("Policies are the same:", np.all(policy1 == policy2))