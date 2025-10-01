import numpy as np

# -------------------------------
# Sample dataset (tiny sentiment)
# -------------------------------
sentences = [
    "I love this movie",
    "This film is great",
    "What a fantastic experience",
    "I really enjoyed the show",
    "Amazing performance and story",
    "I hate this movie",
    "This film is terrible",
    "What a boring experience",
    "I really disliked the show",
    "Awful performance and story"
]
labels = np.array([[1,1,1,1,1, 0,0,0,0,0]]).T  # 1=positive, 0=negative

# -------------------------------
# Preprocessing: bag-of-words
# -------------------------------
def tokenize(s):
    return s.lower().split()

# Build vocab
vocab = sorted(set(word for s in sentences for word in tokenize(s)))
vocab_size = len(vocab)
word2idx = {w:i for i,w in enumerate(vocab)}

# Encode sentences as bag-of-words vectors
def encode(s):
    vec = np.zeros(vocab_size)
    for w in tokenize(s):
        if w in word2idx:
            vec[word2idx[w]] += 1
    return vec

X = np.array([encode(s) for s in sentences])

# -------------------------------
# Network forward & unpack
# -------------------------------
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1.0 / (1.0 + np.exp(-x))

def unpack_params(flat, hidden=6):
    # syn0 (input -> hidden)
    syn0 = flat[:vocab_size*hidden].reshape((vocab_size, hidden))
    # syn1 (hidden -> output)
    syn1 = flat[vocab_size*hidden:].reshape((hidden, 1))
    return syn0, syn1

def forward_loss(flat_params, hidden=6):
    syn0, syn1 = unpack_params(flat_params, hidden)
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    mse = np.mean((labels - l2)**2)
    return mse

def forward(flat_params, hidden=6):
    syn0, syn1 = unpack_params(flat_params, hidden)
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    return l0, l1, l2

# -------------------------------
# Wolf-Pack Hunting + Relativity
# -------------------------------
def wolf_pack_optimize(
    dim,
    fitness_fn,
    pop_size=30,
    iterations=2000,
    c1=1.5, c2=1.5, c3=0.5,
    inertia=0.6,
    c_max_speed=1.0,
    scout_prob=0.05,
    seed=1
):
    rng = np.random.RandomState(seed)
    positions = rng.uniform(-1, 1, size=(pop_size, dim))
    velocities = rng.normal(scale=0.1, size=(pop_size, dim))
    fitness = np.array([fitness_fn(p) for p in positions])

    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_fit = fitness[best_idx]

    for it in range(1, iterations+1):
        order = np.argsort(fitness)
        alpha_pos = positions[order[0]].copy()
        beta_pos  = positions[order[1]].copy() if pop_size>1 else alpha_pos.copy()
        delta_pos = positions[order[2]].copy() if pop_size>2 else alpha_pos.copy()

        for i in range(pop_size):
            pos = positions[i]
            vel = velocities[i]

            A1 = c1 * rng.rand(dim)
            A2 = c2 * rng.rand(dim)
            A3 = c3 * rng.rand(dim)

            acc = (A1 * (alpha_pos - pos)
                   + A2 * (beta_pos - pos)
                   + A3 * (delta_pos - pos))

            vel = inertia * vel + acc

            if rng.rand() < scout_prob:
                vel += rng.normal(scale=0.5, size=dim)

            speed = np.linalg.norm(vel)
            c = c_max_speed

            if speed >= c:
                vel = vel / speed * (c * (1 - 1e-6))
                speed = np.linalg.norm(vel)

            beta = speed / c
            eps = 1e-12
            gamma = 1.0 / (np.sqrt(max(1.0 - beta**2, eps)))

            step = vel / gamma
            new_pos = pos + step
            new_pos = np.clip(new_pos, -5.0, 5.0)

            positions[i] = new_pos
            velocities[i] = vel

        for i in range(pop_size):
            fitness[i] = fitness_fn(positions[i])

        curr_best_idx = np.argmin(fitness)
        curr_best_fit = fitness[curr_best_idx]
        if curr_best_fit < best_fit:
            best_fit = curr_best_fit
            best_pos = positions[curr_best_idx].copy()

        if it % 200 == 0:
            worst_idx = np.argmax(fitness)
            positions[worst_idx] = best_pos + rng.normal(scale=0.1, size=dim)
            velocities[worst_idx] = rng.normal(scale=0.05, size=dim)
            fitness[worst_idx] = fitness_fn(positions[worst_idx])

        if it % 100 == 0 or it == 1:
            print(f"Iter {it:5d}  best MSE = {best_fit:.6f}")

    return best_pos, best_fit

# -------------------------------
# Run training
# -------------------------------
if __name__ == "__main__":
    hidden = 6
    dim = vocab_size * hidden + hidden * 1  # syn0 + syn1
    best_flat, best_mse = wolf_pack_optimize(
        dim=dim,
        fitness_fn=lambda p: forward_loss(p, hidden),
        pop_size=40,
        iterations=1500,
        seed=42
    )

    print("\nBest MSE found:", best_mse)
    _, _, preds = forward(best_flat, hidden)
    print("\nPredictions on training sentences:")
    for s, p, y_true in zip(sentences, preds, labels):
        print(f"{s:35s} -> pred={p[0]:.3f}  true={y_true[0]}")
