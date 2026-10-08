import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

H, W, MINES = 9, 9, 10
HIDDEN, OFF_BOARD, KNOWN_MINE = -1, -2, 9


def neighbours(r, c):
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr != 0 or dc != 0) and 0 <= r + dr < H and 0 <= c + dc < W:
                out.append((r + dr, c + dc))
    return out


class Game:
    def __init__(self, rng):
        self.rng = rng
        self.mines = None
        self.counts = None
        self.view = np.full((H, W), HIDDEN, dtype=np.int8)
        self.alive = True

    def won(self):
        return self.alive and (self.view == HIDDEN).sum() == MINES

    def place_mines(self, r, c):
        # mines go down after the first click so it is always safe
        banned = [(r, c)] + neighbours(r, c)
        free = [(i, j) for i in range(H) for j in range(W) if (i, j) not in banned]
        picks = self.rng.choice(len(free), size=MINES, replace=False)
        self.mines = np.zeros((H, W), dtype=bool)
        for k in picks:
            self.mines[free[k]] = True
        self.counts = np.zeros((H, W), dtype=np.int8)
        for i in range(H):
            for j in range(W):
                self.counts[i, j] = sum(self.mines[p] for p in neighbours(i, j))

    def reveal(self, r, c):
        if self.mines is None:
            self.place_mines(r, c)
        if self.view[r, c] != HIDDEN:
            return
        if self.mines[r, c]:
            self.alive = False
            return
        stack = [(r, c)]
        while stack:
            i, j = stack.pop()
            if self.view[i, j] != HIDDEN:
                continue
            self.view[i, j] = self.counts[i, j]
            if self.counts[i, j] == 0:
                for p in neighbours(i, j):
                    if self.view[p] == HIDDEN:
                        stack.append(p)


def logic_moves(view):
    # find cells that are definitely safe / definitely mines
    safe, mines = set(), set()
    changed = True
    while changed:
        changed = False
        for r in range(H):
            for c in range(W):
                n = view[r, c]
                if n <= 0:
                    continue
                known = 0
                unknown = []
                for p in neighbours(r, c):
                    if p in mines:
                        known += 1
                    elif view[p] == HIDDEN and p not in safe:
                        unknown.append(p)
                if len(unknown) == 0:
                    continue
                left = n - known
                if left == 0:
                    safe.update(unknown)
                    changed = True
                elif left == len(unknown):
                    mines.update(unknown)
                    changed = True
    return safe, mines


def get_features(view, known_mines, cells):
    v = view.astype(np.int16)
    for p in known_mines:
        v[p] = KNOWN_MINE
    padded = np.pad(v, 2, constant_values=OFF_BOARD)
    hidden = (view == HIDDEN).sum()
    density = (MINES - len(known_mines)) / max(hidden - len(known_mines), 1)
    rows = []
    for r, c in cells:
        window = padded[r:r + 5, c:c + 5].ravel()
        window = np.delete(window, 12) + 2  # drop the centre, shift codes to 0..11
        rows.append(np.concatenate([window, [density, hidden / (H * W)]]))
    return np.array(rows, dtype=np.float32)


def candidates(view, known_mines):
    return [(r, c) for r in range(H) for c in range(W)
            if view[r, c] == HIDDEN and (r, c) not in known_mines]


def make_training_data(n_games, seed):
    X, y, game_ids, state_ids = [], [], [], []
    state = 0
    for g in range(n_games):
        rng = np.random.default_rng([seed, g])
        game = Game(rng)
        game.reveal(int(rng.integers(H)), int(rng.integers(W)))
        while game.alive and not game.won():
            safe, mines = logic_moves(game.view)
            if safe:
                for p in safe:
                    game.reveal(*p)
                continue
            # stuck: save every candidate cell with its true label
            cells = candidates(game.view, mines)
            X.append(get_features(game.view, mines, cells))
            y.extend(int(game.mines[p]) for p in cells)
            game_ids.extend([g] * len(cells))
            state_ids.extend([state] * len(cells))
            state += 1
            # keep the game going using a cell we know is safe
            safe_cells = [p for p in cells if not game.mines[p]]
            game.reveal(*safe_cells[int(rng.integers(len(safe_cells)))])
    return np.vstack(X), np.array(y), np.array(game_ids), np.array(state_ids)


def play(board_seed, policy, model=None):
    rng = np.random.default_rng(board_seed)
    guess_rng = np.random.default_rng(board_seed + 1_000_003)
    game = Game(rng)
    game.reveal(int(rng.integers(H)), int(rng.integers(W)))
    guesses = 0
    while game.alive and not game.won():
        safe, mines = logic_moves(game.view)
        if safe:
            for p in safe:
                game.reveal(*p)
            continue
        cells = candidates(game.view, mines)
        guesses += 1
        if policy == "random":
            cell = cells[int(guess_rng.integers(len(cells)))]
        else:
            probs = model.predict_proba(get_features(game.view, mines, cells))[:, 1]
            cell = cells[int(np.argmin(probs))]
        game.reveal(*cell)
    cleared = (game.view != HIDDEN).sum() / (H * W - MINES)
    return game.won(), guesses, cleared


def evaluate(policy, model, n_games, seed):
    results = [play(seed * 1_000_000 + i, policy, model) for i in range(n_games)]
    won = np.array([r[0] for r in results])
    guesses = np.array([r[1] for r in results])
    cleared = np.array([r[2] for r in results])
    p = won.mean()
    return {"win_rate": p,
            "ci95": 1.96 * np.sqrt(p * (1 - p) / n_games),
            "avg_cleared": cleared.mean(),
            "avg_guesses": guesses.mean(),
            "no_guess_wins": float((won & (guesses == 0)).mean())}


def guess_quality(model, X, y, state_ids):
    # how often is the model's lowest-risk cell actually safe, vs a random cell?
    df = pd.DataFrame({"s": state_ids, "p": model.predict_proba(X)[:, 1], "y": y})
    best = df.loc[df.groupby("s")["p"].idxmin()]
    random_safe = 1 - df.groupby("s")["y"].mean().mean()
    return 1 - best["y"].mean(), random_safe


def main(train_games=6000, eval_games=3000, seed=0):
    print("making training data...")
    X, y, game_ids, state_ids = make_training_data(train_games, seed)
    print(len(y), "cells,", len(np.unique(state_ids)), "stuck states, mine rate", round(y.mean(), 3))

    # split by game so no board is in both train and test
    test = game_ids >= int(0.8 * train_games)
    X_train, y_train, X_test, y_test = X[~test], y[~test], X[test], y[test]

    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           max_leaf_nodes=31,
                                           categorical_features=list(range(24)),
                                           random_state=seed)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    baseline = np.full_like(probs, y_train.mean())
    print("AUC", round(roc_auc_score(y_test, probs), 3))
    print("log loss", round(log_loss(y_test, probs), 3), "vs baseline", round(log_loss(y_test, baseline), 3))
    best, rand = guess_quality(model, X_test, y_test, state_ids[test])
    print("guess is safe:", round(best, 3), "(model) vs", round(rand, 3), "(random)")

    print("playing full games...")
    table = {}
    table["Logic + random guess"] = evaluate("random", model, eval_games, seed + 7)
    table["Logic + ML guess"] = evaluate("ml", model, eval_games, seed + 7)
    res = pd.DataFrame(table).T
    print(res.round(3).to_string())
    res.round(4).to_csv("results.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(res.index, res["win_rate"] * 100, yerr=res["ci95"] * 100, capsize=6)
    ax[0].set_ylabel("win rate (%)")
    ax[0].set_title("win rate over full games")
    ax[1].bar(["random cell", "model's pick"], [rand * 100, best * 100])
    ax[1].set_ylabel("guess was safe (%)")
    ax[1].set_title("guess quality (held-out)")
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)


if __name__ == "__main__":
    # optional: python minesweeper_simple.py 3000 1000 1
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
