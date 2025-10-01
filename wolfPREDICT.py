import numpy as np

# ---------- Problem (XOR-like toy) ----------
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# ---------- Network forward & unflatten helpers ----------
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1.0 / (1.0 + np.exp(-x))

def unpack_params(flat):
    # shapes: syn0 (3 x 4) -> 12 params, syn1 (4 x 1) -> 4 params => total 16
    syn0 = flat[:12].reshape((3,4))
    syn1 = flat[12:16].reshape((4,1))
    return syn0, syn1

def forward_loss(flat_params):
    syn0, syn1 = unpack_params(flat_params)
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    mse = np.mean((y - l2)**2)
    return mse

# Optional: return output too for inspection
def forward(flat_params):
    syn0, syn1 = unpack_params(flat_params)
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    return l0, l1, l2

# ---------- Wolf-Pack Hunting + "Relativistic" velocity update ----------
def wolf_pack_optimize(
    dim,
    fitness_fn,
    pop_size=30,
    iterations=2000,
    c1=1.5, c2=1.5, c3=0.5,   # social coefficients
    inertia=0.6,              # velocity inertia term
    c_max_speed=1.0,          # relativistic "speed of light" cap
    scout_prob=0.05,          # random scouting prob
    seed=1
):
    rng = np.random.RandomState(seed)
    # initialize positions and velocities
    # positions in [-1,1]
    positions = rng.uniform(-1, 1, size=(pop_size, dim))
    velocities = rng.normal(scale=0.1, size=(pop_size, dim))
    fitness = np.array([fitness_fn(p) for p in positions])

    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_fit = fitness[best_idx]

    for it in range(1, iterations+1):
        # sort wolves by fitness (ascending)
        order = np.argsort(fitness)
        alpha_pos = positions[order[0]].copy()
        beta_pos  = positions[order[1]].copy() if pop_size>1 else alpha_pos.copy()
        delta_pos = positions[order[2]].copy() if pop_size>2 else alpha_pos.copy()

        for i in range(pop_size):
            pos = positions[i]
            vel = velocities[i]

            # social attraction to alpha/beta/delta (pack hunting)
            r1, r2, r3 = rng.rand(3), rng.rand(3), rng.rand(3)
            # we use vector weights per-dimension using random scalars
            A1 = c1 * rng.rand(dim)
            A2 = c2 * rng.rand(dim)
            A3 = c3 * rng.rand(dim)

            # acceleration-like terms toward leaders
            acc = (A1 * (alpha_pos - pos)
                   + A2 * (beta_pos - pos)
                   + A3 * (delta_pos - pos))

            # velocity update with inertia
            vel = inertia * vel + acc

            # scout behavior: occasional random exploration
            if rng.rand() < scout_prob:
                vel += rng.normal(scale=0.5, size=dim)

            # ---- Special relativity inspired handling ----
            # speed = ||v|| (Euclidean)
            speed = np.linalg.norm(vel)
            c = c_max_speed

            # if speed approaches c, cap and compute Lorentz gamma
            if speed >= c:
                # scale velocity down to just under c
                vel = vel / speed * (c * (1 - 1e-6))
                speed = np.linalg.norm(vel)

            # Lorentz factor gamma for speed ratio beta = v/c
            beta = speed / c
            # Numerical safety for gamma
            eps = 1e-12
            if beta >= 1.0:
                gamma = 1e6
            else:
                gamma = 1.0 / (np.sqrt(max(1.0 - beta**2, eps)))

            # Effective step scaled by time dilation: we divide by gamma
            step = vel / gamma

            # update position
            new_pos = pos + step

            # boundary handling (keep in reasonable bounds)
            new_pos = np.clip(new_pos, -5.0, 5.0)

            positions[i] = new_pos
            velocities[i] = vel

        # evaluate fitness
        for i in range(pop_size):
            fitness[i] = fitness_fn(positions[i])

        # update best
        curr_best_idx = np.argmin(fitness)
        curr_best_fit = fitness[curr_best_idx]
        if curr_best_fit < best_fit:
            best_fit = curr_best_fit
            best_pos = positions[curr_best_idx].copy()

        # optional: pack behaviors - reinitialize worst wolves near best occasionally (besieging)
        if it % 200 == 0:
            worst_idx = np.argmax(fitness)
            # place worst around alpha with small random offset
            positions[worst_idx] = best_pos + rng.normal(scale=0.1, size=dim)
            velocities[worst_idx] = rng.normal(scale=0.05, size=dim)
            fitness[worst_idx] = fitness_fn(positions[worst_idx])

        if it % 100 == 0 or it == 1:
            print(f"Iter {it:5d}  best MSE = {best_fit:.6f}")

    return best_pos, best_fit

# ---------- Run optimizer ----------
if __name__ == "__main__":
    dim = 16  # 12 + 4 parameters for syn0 & syn1
    np.random.seed(1)

    best_flat, best_mse = wolf_pack_optimize(
        dim=dim,
        fitness_fn=forward_loss,
        pop_size=40,
        iterations=2500,
        c1=1.4, c2=1.4, c3=0.6,
        inertia=0.65,
        c_max_speed=1.5,
        scout_prob=0.07,
        seed=42
    )

    print("\nBest MSE found:", best_mse)
    syn0_best, syn1_best = unpack_params(best_flat)
    print("syn0 shape:", syn0_best.shape, " syn1 shape:", syn1_best.shape)

    # inspect outputs on dataset
    _, _, l2 = forward(best_flat)
    print("Outputs (predictions):")
    print(l2)
    print("Rounded predictions:")
    print(np.round(l2))

    # Optional: small local backprop refinement (few gradient steps)
    # If you want both global search + local tuning, try a few backprop updates:
    def backprop_refine(flat_params, steps=500, lr=0.5):
        syn0, syn1 = unpack_params(flat_params)
        for _ in range(steps):
            # forward
            l0 = X
            l1 = sigmoid(np.dot(l0, syn0))
            l2 = sigmoid(np.dot(l1, syn1))

            l2_error = y - l2
            l2_delta = l2_error * sigmoid(l2, deriv=True)
            l1_error = l2_delta.dot(syn1.T)
            l1_delta = l1_error * sigmoid(l1, deriv=True)

            syn1 += lr * l1.T.dot(l2_delta)
            syn0 += lr * l0.T.dot(l1_delta)
        return np.concatenate([syn0.ravel(), syn1.ravel()])

    print("\nRefining best solution with local backprop (optional)...")
    refined_flat = backprop_refine(best_flat.copy(), steps=400, lr=0.35)
    print("Refined MSE:", forward_loss(refined_flat))
    _, _, l2r = forward(refined_flat)
    print("Refined outputs:")
    print(l2r)
    print("Rounded refined predictions:")
    print(np.round(l2r))
