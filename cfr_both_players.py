import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import lru_cache
from efg_best_response import parse_game, Game

Behavior = dict


def eval_value(game, sigma1, sigma2):
    @lru_cache(None)
    def V(h):
        node = game.nodes[h]
        if node.kind == "terminal":
            return node.payoffs[1]
        if node.kind == "chance":
            return sum(
                p * V(game.children_hist(node, a))
                for a, p in node.chance_actions.items()
            )
        I = node.infoset
        strat = sigma1[I] if node.player == 1 else sigma2[I]
        return sum(p * V(game.children_hist(node, a)) for a, p in strat.items())

    return V(game.root)


def best_response_value(game, br_player, opponent_sigma):
    @lru_cache(None)
    def Vdesc(h):
        node = game.nodes[h]
        if node.kind == "terminal":
            return node.payoffs[1]
        if node.kind == "chance":
            return sum(
                p * Vdesc(game.children_hist(node, a))
                for a, p in node.chance_actions.items()
            )
        if node.player == br_player:
            vals = [Vdesc(game.children_hist(node, a)) for a in node.actions]
            return max(vals) if br_player == 1 else min(vals)
        I = node.infoset
        return sum(
            p * Vdesc(game.children_hist(node, a)) for a, p in opponent_sigma[I].items()
        )

    Q = defaultdict(lambda: defaultdict(float))

    def collect(h, w):
        node = game.nodes[h]
        if node.kind == "terminal":
            return
        if node.kind == "chance":
            for a, p in node.chance_actions.items():
                collect(game.children_hist(node, a), w * p)
            return
        if node.player == br_player:
            I = node.infoset
            for a in node.actions:
                Q[I][a] += w * Vdesc(game.children_hist(node, a))
            for a in node.actions:
                collect(game.children_hist(node, a), w)
            return
        I = node.infoset
        for a, p in opponent_sigma[I].items():
            collect(game.children_hist(node, a), w * p)

    collect(game.root, 1.0)

    br_sigma = {}
    for I, acts_dict in Q.items():
        if br_player == 1:
            a_star = max(acts_dict, key=lambda a: acts_dict[a])
        else:
            a_star = min(acts_dict, key=lambda a: acts_dict[a])
        br_sigma[I] = {a: (1.0 if a == a_star else 0.0) for a in acts_dict}

    @lru_cache(None)
    def Vfix(h):
        node = game.nodes[h]
        if node.kind == "terminal":
            return node.payoffs[1]
        if node.kind == "chance":
            return sum(
                p * Vfix(game.children_hist(node, a))
                for a, p in node.chance_actions.items()
            )
        I = node.infoset
        if node.player == br_player:
            a = max(br_sigma[I], key=lambda a: br_sigma[I][a])
            return Vfix(game.children_hist(node, a))
        strat = opponent_sigma[I]
        return sum(p * Vfix(game.children_hist(node, a)) for a, p in strat.items())

    return Vfix(game.root)


class CFRBoth:
    def __init__(self, game):
        self.g = game
        self.R = defaultdict(lambda: defaultdict(float))
        self.S = defaultdict(lambda: defaultdict(float))

    def regret_match(self, I):
        acts = list(self.S[I].keys())
        pos = np.array([max(self.R[I][a], 0.0) for a in acts], float)
        if pos.sum() == 0:
            pos[:] = 1.0 / len(pos)
        else:
            pos /= pos.sum()
        return {a: p for a, p in zip(acts, pos.tolist())}

    def avg_strategy(self):
        avg = {}
        for I in self.S:
            total = sum(self.S[I].values())
            if total > 0:
                avg[I] = {a: self.S[I][a] / total for a in self.S[I]}
            else:
                k = len(self.S[I])
                avg[I] = {a: 1.0 / k for a in self.S[I]}
        return avg

    def cfr(self, h, r1, r2):
        node = self.g.nodes[h]
        if node.kind == "terminal":
            return node.payoffs[1]
        if node.kind == "chance":
            return sum(
                p * self.cfr(self.g.children_hist(node, a), r1, r2)
                for a, p in node.chance_actions.items()
            )
        I = node.infoset
        for a in node.actions:
            self.S[I].setdefault(a, 0.0)
        sigma = self.regret_match(I)
        vals, ev = [], 0.0

        if node.player == 1:
            for a in node.actions:
                pa = sigma[a]
                Qa = self.cfr(self.g.children_hist(node, a), r1 * pa, r2)
                vals.append(Qa)
                ev += pa * Qa
            if r2 > 0:
                for a, Qa in zip(node.actions, vals):
                    self.R[I][a] += r2 * (Qa - ev)
            if r1 > 0:
                for a in node.actions:
                    self.S[I][a] += r1 * sigma[a]
            return ev
        else:
            for a in node.actions:
                pa = sigma[a]
                Qa = self.cfr(self.g.children_hist(node, a), r1, r2 * pa)
                vals.append(Qa)
                ev += pa * Qa
            if r1 > 0:
                for a, Qa in zip(node.actions, vals):
                    self.R[I][a] += r1 * ((-Qa) - (-ev))
            if r2 > 0:
                for a in node.actions:
                    self.S[I][a] += r2 * sigma[a]
            return ev


def run_single(gamefile, iters=1000):
    print(f"\n=== Running CFR on {gamefile} ===")
    game = parse_game(gamefile)
    cfr = CFRBoth(game)

    utils, gaps = [], []

    for t in range(1, iters + 1):
        cfr.cfr(game.root, 1.0, 1.0)
        sigma = cfr.avg_strategy()

        # utility
        sigma1 = sigma
        sigma2 = sigma
        u1 = eval_value(game, sigma1, sigma2)
        utils.append(u1)

        # Nash gap γ = BR1 - BR2
        max_u1 = best_response_value(game, 1, sigma2)
        min_u1 = best_response_value(game, 2, sigma1)
        gaps.append(max_u1 - min_u1)

    # # Plot Utility
    # plt.figure()
    # plt.plot(utils)
    # plt.title(f"P1 Utility vs Iterations — {gamefile}")
    # plt.xlabel("Iteration")
    # plt.ylabel("u1(x̄,ȳ)")
    # plt.grid(True)

    # # Plot Nash Gap
    # plt.figure()
    # plt.plot(gaps)
    # plt.title(f"Nash Gap vs Iterations — {gamefile}")
    # plt.xlabel("Iteration")
    # plt.ylabel("γ(x̄,ȳ)")
    # plt.grid(True)


def run_all_iters(gamefile, iters=1000):
    game = parse_game(gamefile)
    cfr = CFRBoth(game)
    utils, gaps = [], []

    for t in range(1, iters + 1):
        cfr.cfr(game.root, 1.0, 1.0)
        sigma = cfr.avg_strategy()

        def fix_sigma(player):
            out = {}
            for h, node in game.nodes.items():
                if node.kind != "player" or node.player != player:
                    continue
                I = node.infoset
                if I not in sigma:
                    k = len(node.actions)
                    out[I] = {a: 1.0 / k for a in node.actions}
                else:
                    out[I] = sigma[I]
            return out

        sigma1 = fix_sigma(1)
        sigma2 = fix_sigma(2)

        utils.append(eval_value(game, sigma1, sigma2))
        max_u1 = best_response_value(game, 1, sigma2)
        min_u1 = best_response_value(game, 2, sigma1)
        gaps.append(max_u1 - min_u1)

    return np.array(utils), np.array(gaps)


def run_all():
    games = [
        ("rock_paper_superscissors.txt", "RPSS"),
        ("kuhn.txt", "Kuhn Poker"),
        ("leduc2.txt", "Leduc Poker"),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), squeeze=False)

    for r, (fname, title) in enumerate(games):
        utils, gaps = run_all_iters(fname, iters=1000)
        x = np.arange(1, len(utils) + 1)

        axes[r, 0].plot(x, utils)
        axes[r, 0].set_title(f"{title}: Utility")
        axes[r, 0].set_xlabel("Iterations")
        axes[r, 0].set_ylabel("u1")
        axes[r, 0].grid(True)

        axes[r, 1].plot(x, gaps)
        axes[r, 1].set_title(f"{title}: Nash Gap")
        axes[r, 1].set_xlabel("Iterations")
        axes[r, 1].set_ylabel("γ")
        axes[r, 1].grid(True)

    fig.suptitle(
        "CFR Performance on 3 Extensive-Form Games (1000 Iterations Each)", fontsize=16
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    run_all()


if __name__ == "__main__":
    run_all()
