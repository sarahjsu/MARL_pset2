import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from efg_best_response import parse_game, uniform_strategy, value_u1, Game, Behavior

Behavior = dict


class CFR:
    def __init__(self, game):
        self.game = game
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

    def _regret_match(self, I):
        acts = self.game.infoset_actions[I]
        pos = np.array([max(self.regret_sum[I][a], 0.0) for a in acts], dtype=float)
        s = pos.sum()
        if s > 0:
            pos /= s
        else:
            pos[:] = 1.0 / len(acts)
        return {a: p for a, p in zip(acts, pos.tolist())}

    def _avg_strategy(self):
        avg = {}
        for I, acts in self.game.infoset_actions.items():
            if self.game.infoset_owner[I] != 1:
                continue
            total = sum(self.strategy_sum[I][a] for a in acts)
            if total > 0:
                avg[I] = {a: self.strategy_sum[I][a] / total for a in acts}
            else:
                k = len(acts)
                avg[I] = {a: 1.0 / k for a in acts}
        return avg

    def _cfr_iter(self, h, reach_p1, reach_p2, sigma2_uniform):
        node = self.game.nodes[h]

        if node.kind == "terminal":
            return node.payoffs[1]

        if node.kind == "chance":
            ev = 0.0
            for a, p in node.chance_actions.items():
                ev += p * self._cfr_iter(
                    self.game.children_hist(node, a), reach_p1, reach_p2, sigma2_uniform
                )
            return ev

        if node.player == 2:
            I = node.infoset
            ev = 0.0
            for a, p in sigma2_uniform[I].items():
                ev += p * self._cfr_iter(
                    self.game.children_hist(node, a), reach_p1, reach_p2, sigma2_uniform
                )
            return ev

        I = node.infoset
        acts = self.game.infoset_actions[I]
        sigma1 = self._regret_match(I)

        action_vals = []
        node_ev = 0.0
        for a in acts:
            pa = sigma1[a]
            Qa = self._cfr_iter(
                self.game.children_hist(node, a),
                reach_p1 * pa,
                reach_p2,
                sigma2_uniform,
            )
            action_vals.append(Qa)
            node_ev += pa * Qa

        if reach_p2 > 0.0:
            for a, Qa in zip(acts, action_vals):
                self.regret_sum[I][a] += reach_p2 * (Qa - node_ev)

        if reach_p1 > 0.0:
            for a in acts:
                self.strategy_sum[I][a] += reach_p1 * sigma1[a]

        return node_ev

    def _eval_avg(self, h, sigma2_uniform, sigma1_avg):
        node = self.game.nodes[h]
        if node.kind == "terminal":
            return node.payoffs[1]
        if node.kind == "chance":
            s = 0.0
            for a, p in node.chance_actions.items():
                s += p * self._eval_avg(
                    self.game.children_hist(node, a), sigma2_uniform, sigma1_avg
                )
            return s
        if node.player == 2:
            I = node.infoset
            s = 0.0
            for a, p in sigma2_uniform[I].items():
                s += p * self._eval_avg(
                    self.game.children_hist(node, a), sigma2_uniform, sigma1_avg
                )
            return s
        else:
            I = node.infoset
            s = 0.0
            for a, p in sigma1_avg[I].items():
                s += p * self._eval_avg(
                    self.game.children_hist(node, a), sigma2_uniform, sigma1_avg
                )
            return s


def run(gamefile, iters=1000):
    game = parse_game(gamefile)
    cfr = CFR(game)
    sigma2U = uniform_strategy(game, 2)

    utilities = []
    for t in range(1, iters + 1):
        cfr._cfr_iter(game.root, 1.0, 1.0, sigma2U)
        sigma1_avg = cfr._avg_strategy()
        utilities.append(cfr._eval_avg(game.root, sigma2U, sigma1_avg))

    plt.plot(range(1, iters + 1), utilities)
    plt.xlabel("Iterations")
    plt.ylabel("Player 1 EV (avg strat vs uniform P2)")
    plt.title(f"CFR (P1 only) — {gamefile}")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run(sys.argv[1], iters)
