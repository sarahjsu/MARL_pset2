from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
import math
import re
from collections import defaultdict

History = str
Action = str


@dataclass
class Node:
    history: History
    kind: str
    player: Optional[int] = None
    actions: List[Action] = field(default_factory=list)
    chance_actions: Dict[Action, float] = field(default_factory=dict)
    payoffs: Dict[int, float] = field(default_factory=dict)
    infoset: Optional[str] = None


@dataclass
class Game:
    nodes: Dict[History, Node]
    infosets: Dict[str, List[History]]
    infoset_owner: Dict[str, int]
    infoset_actions: Dict[str, List[Action]]
    root: History

    def children_hist(self, node, a):
        if node.kind == "chance":
            mover = "C"
        elif node.kind == "player":
            assert node.player in (1, 2)
            mover = f"P{node.player}"
        else:
            raise ValueError("Terminal nodes have no children.")
        base = node.history
        if base == "":
            base = "/"
        if not base.endswith("/"):
            base = base + "/"
        return f"{base}{mover}:{a}/"


def parse_game(path):
    nodes: Dict[History, Node] = {}
    infosets: Dict[str, List[History]] = defaultdict(list)

    def tokify(s):
        return re.findall(r"[^\s]+", s.strip())

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = tokify(line)
            if toks[0].lower() == "node":
                history = toks[1]
                kind = toks[2].lower()
                if kind == "player":
                    player = int(toks[3])
                    assert toks[4].lower() == "actions"
                    actions = toks[5:]
                    nodes[history] = Node(
                        history=history, kind="player", player=player, actions=actions
                    )
                elif kind == "chance":
                    assert toks[3].lower() == "actions"
                    pairs = toks[4:]
                    cad: Dict[str, float] = {}
                    for p in pairs:
                        a, pr = p.split("=")
                        cad[a] = float(pr)
                    if abs(sum(cad.values()) - 1.0) > 1e-6:
                        raise ValueError(f"Chance probs at {history} don't sum to 1.")
                    nodes[history] = Node(
                        history=history, kind="chance", chance_actions=cad
                    )
                elif kind == "terminal":
                    assert toks[3].lower() == "payoffs"
                    pays = {}
                    for pair in toks[4:]:
                        pid, val = pair.split("=")
                        pays[int(pid)] = float(val)
                    if 1 not in pays or 2 not in pays:
                        raise ValueError(f"Terminal missing payoffs at {history}")
                    nodes[history] = Node(
                        history=history, kind="terminal", payoffs=pays
                    )
                else:
                    raise ValueError(f"Unknown node kind: {kind}")

            elif toks[0].lower() == "infoset":
                name = toks[1]
                assert toks[2].lower() == "nodes"
                hists = toks[3:]
                infosets[name].extend(hists)
            else:
                raise ValueError(f"Unknown line start: {toks[0]}")

    infoset_owner: Dict[str, int] = {}
    infoset_actions: Dict[str, List[str]] = {}
    for name, hs in infosets.items():
        if not hs:
            continue
        first = nodes[hs[0]]
        if first.kind != "player":
            raise ValueError(f"Infoset {name} contains non-player nodes")
        owner = first.player
        acts = first.actions[:]
        for h in hs:
            n = nodes[h]
            n.infoset = name
            if n.kind != "player" or n.player != owner:
                raise ValueError(f"Inconsistent infoset {name}")
            if n.actions != acts:
                raise ValueError(f"Actions differ inside infoset {name}")
        infoset_owner[name] = owner
        infoset_actions[name] = acts

    root = "/"
    if root not in nodes:
        if "" in nodes:
            root = ""
        else:
            root = min(nodes.keys(), key=len)

    return Game(
        nodes=nodes,
        infosets=infosets,
        infoset_owner=infoset_owner,
        infoset_actions=infoset_actions,
        root=root,
    )


Behavior = Dict[str, Dict[str, float]]


def uniform_strategy(game, player):
    sigma: Behavior = {}
    for I, owner in game.infoset_owner.items():
        if owner != player:
            continue
        acts = game.infoset_actions[I]
        p = 1.0 / len(acts)
        sigma[I] = {a: p for a in acts}
    return sigma


def value_u1(game, sigma1, sigma2):
    memo: Dict[History, float] = {}

    def v(h):
        if h in memo:
            return memo[h]
        node = game.nodes[h]
        if node.kind == "terminal":
            memo[h] = node.payoffs[1]
            return memo[h]
        if node.kind == "chance":
            s = 0.0
            for a, p in node.chance_actions.items():
                s += p * v(game.children_hist(node, a))
            memo[h] = s
            return s
        if node.player == 1:
            if sigma1 is None:
                raise RuntimeError("sigma1 is None but needed in evaluation.")
            I = node.infoset
            probs = sigma1[I]
            s = 0.0
            for a, p in probs.items():
                s += p * v(game.children_hist(node, a))
            memo[h] = s
            return s
        else:
            if sigma2 is None:
                raise RuntimeError("sigma2 is None but needed in evaluation.")
            I = node.infoset
            probs = sigma2[I]
            s = 0.0
            for a, p in probs.items():
                s += p * v(game.children_hist(node, a))
            memo[h] = s
            return s

    return v(game.root)


def best_response_to(game, br_player, opponent_sigma):
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best_descendant_value(h):
        node = game.nodes[h]
        if node.kind == "terminal":
            return node.payoffs[1]
        if node.kind == "chance":
            return sum(
                p * best_descendant_value(game.children_hist(node, a))
                for a, p in node.chance_actions.items()
            )
        if node.player == br_player:
            child_vals = [
                best_descendant_value(game.children_hist(node, a)) for a in node.actions
            ]
            return (max if br_player == 1 else min)(child_vals)
        else:
            I = node.infoset
            probs = opponent_sigma[I]
            return sum(
                p * best_descendant_value(game.children_hist(node, a))
                for a, p in probs.items()
            )

    Q_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def accumulate(h, w_opp_ch):
        node = game.nodes[h]
        if node.kind == "terminal":
            return
        if node.kind == "chance":
            for a, p in node.chance_actions.items():
                accumulate(game.children_hist(node, a), w_opp_ch * p)
            return
        if node.player == br_player:
            I = node.infoset
            for a in node.actions:
                child = game.children_hist(node, a)
                Q_sum[I][a] += w_opp_ch * best_descendant_value(child)
            for a in node.actions:
                accumulate(game.children_hist(node, a), w_opp_ch)
            return
        else:
            I = node.infoset
            for a, p in opponent_sigma[I].items():
                accumulate(game.children_hist(node, a), w_opp_ch * p)

    accumulate(game.root, 1.0)

    br_behavior: Behavior = {}
    for I, acts in game.infoset_actions.items():
        if game.infoset_owner[I] != br_player:
            continue
        if not Q_sum[I]:
            a_star = acts[0]
        else:
            scored = [(Q_sum[I][a], a) for a in acts]
            if br_player == 1:
                a_star = max(scored)[1]
            else:
                a_star = min(scored)[1]
        br_behavior[I] = {a: (1.0 if a == a_star else 0.0) for a in acts}

    def eval_fixed(h):
        node = game.nodes[h]
        if node.kind == "terminal":
            return node.payoffs[1]
        if node.kind == "chance":
            return sum(
                p * eval_fixed(game.children_hist(node, a))
                for a, p in node.chance_actions.items()
            )
        if node.player == br_player:
            I = node.infoset
            a = max(br_behavior[I], key=lambda x: br_behavior[I][x])
            return eval_fixed(game.children_hist(node, a))
        else:
            I = node.infoset
            probs = opponent_sigma[I]
            return sum(
                p * eval_fixed(game.children_hist(node, a)) for a, p in probs.items()
            )

    root_val = eval_fixed(game.root)
    return root_val, br_behavior


def nash_gap_uniform(game):
    sigma2U = uniform_strategy(game, player=2)
    val_BR1, br1 = best_response_to(game, br_player=1, opponent_sigma=sigma2U)

    sigma1U = uniform_strategy(game, player=1)
    val_vs_BR2, br2 = best_response_to(game, br_player=2, opponent_sigma=sigma1U)

    gamma = val_BR1 - val_vs_BR2
    return val_BR1, val_vs_BR2, gamma


def main():
    game_path = sys.argv[1]
    game = parse_game(game_path)

    sigma2U = uniform_strategy(game, 2)
    val_BR1, br1 = best_response_to(game, br_player=1, opponent_sigma=sigma2U)

    print("Best response of Player 1 against uniform Player 2:")
    for I in sorted(br1.keys()):
        choice = [a for a, p in br1[I].items() if p == 1.0][0]
        print(f"  I={I}: a*={choice}")
    print(f"  Expected utility u1 = {val_BR1:.6f}\n")

    val_BR1U, val_U_BR2, gamma = nash_gap_uniform(game)
    print("Uniform vs Uniform profile:")
    print(f"  max_x u1(x, y_U) = {val_BR1U:.6f}")
    print(f"  min_y u1(x_U, y) = {val_U_BR2:.6f}")
    print(f"  Nash gap gamma   = {gamma:.6f}")


if __name__ == "__main__":
    main()
