import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle, Move, MoveCategory, Pokemon, PokemonType, Status
from poke_env.data import GenData
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player

from .base_environment import BaseShowdownEnv

# Gen 9 type chart, used for every effectiveness lookup
TYPE_CHART = GenData.from_gen(9).type_chart

# ---------------------------------------------------------------------------
# Agent action space (high-level actions, remapped onto the 26 Showdown actions)
#   0-3 : use move in slot i
#   4-7 : use move in slot i and terastallize (plain move if tera unavailable)
#   8   : switch to the best-matchup Pokemon on the bench
# Illegal choices (empty slot, disabled move, switching while trapped, ...) are
# remapped to the legal move with the highest estimated damage, and forced
# switches always go to the best-matchup bench Pokemon.
# ---------------------------------------------------------------------------
N_MOVE_SLOTS = 4
ACTION_TERA_OFFSET = 4
ACTION_SWITCH = 8
ACTION_SIZE = 9

# Showdown action ids (see SinglesEnv.action_to_order)
SD_MOVE_OFFSET = 6
SD_TERA_OFFSET = 22
SD_DEFAULT = -2
              
# ---------------------------------------------------------------------------
# Observation layout (every feature is scaled into [0, 1])
# ---------------------------------------------------------------------------
MOVE_FEATURES = 7  # available, dmg, type mult, status, priority, accuracy, tera dmg
OBS_SIZE = (
    N_MOVE_SLOTS * MOVE_FEATURES  # 28 per-move features
    + 12  # active matchup
    + 4  # best switch-in summary
    + 4  # team totals
    + 4  # flags
)

# ---------------------------------------------------------------------------
# Reward shaping weights (potential-based, see calc_reward)
# ---------------------------------------------------------------------------
HP_WEIGHT = 1.0
FAINT_WEIGHT = 1.0
STATUS_WEIGHT = 0.1
VICTORY_REWARD = 5.0

TEAM_SIZE = 6
RANDOM_ROLL_MEAN = 0.925  # mean of Showdown's 0.85-1.00 damage roll
BOOST_STATS = ("atk", "spa", "spe")


# ===========================================================================
# Damage / matchup helpers
# ===========================================================================
def _boost_multiplier(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def _stat(mon: Pokemon, stat: str, known: bool) -> float:
    """Actual stat for our own Pokemon, otherwise estimated from base stats
    using random-battle spreads (31 IVs, 84 EVs) and the Pokemon's level."""
    value = mon.stats.get(stat) if known and mon.stats else None
    if value is None:
        level = mon.level or 80
        value = (2 * mon.base_stats[stat] + 31 + 21) * level // 100 + 5
    value *= _boost_multiplier(mon.boosts.get(stat, 0))
    if stat == "spe" and mon.status == Status.PAR:
        value *= 0.5
    return float(max(value, 1))


def _max_hp(mon: Pokemon, known: bool) -> float:
    if known and mon.max_hp:
        return float(mon.max_hp)
    level = mon.level or 80
    return float((2 * mon.base_stats["hp"] + 31 + 21) * level // 100 + level + 10)


def _type_multiplier(move_type: PokemonType, defender: Pokemon) -> float:
    try:
        return float(
            move_type.damage_multiplier(
                defender.type_1, defender.type_2, type_chart=TYPE_CHART
            )
        )
    except Exception:
        return 1.0


def _damage_fraction(
    base_power: float,
    move_type: PokemonType,
    physical: bool,
    attacker: Pokemon,
    defender: Pokemon,
    attacker_known: bool,
    defender_known: bool,
    stab: float,
    hits: float = 1.0,
    accuracy: float = 1.0,
    defend_physical: Optional[bool] = None,
) -> float:
    """Expected damage as a fraction of the defender's max HP (simplified
    Gen 9 damage formula: no items, abilities, weather or crits)."""
    if base_power <= 0:
        return 0.0
    if defend_physical is None:
        defend_physical = physical
    atk = _stat(attacker, "atk" if physical else "spa", attacker_known)
    dfn = _stat(defender, "def" if defend_physical else "spd", defender_known)
    level = attacker.level or 80
    base = ((2 * level / 5 + 2) * base_power * atk / dfn) / 50 + 2
    damage = base * stab * _type_multiplier(move_type, defender) * RANDOM_ROLL_MEAN
    if physical and attacker.status == Status.BRN:
        damage *= 0.5
    return damage * hits * accuracy / _max_hp(defender, defender_known)


def _own_tera_type(mon: Pokemon) -> Optional[PokemonType]:
    if mon.tera_type is not None:
        return mon.tera_type
    request = getattr(mon, "_last_request", None) or {}
    tera_name = request.get("teraType")
    if not tera_name:
        return None
    try:
        return PokemonType.from_name(tera_name)
    except Exception:
        return None


def _move_damage(
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
    attacker_known: bool,
    defender_known: bool,
    tera: bool = False,
) -> float:
    """Expected damage of a real move as a fraction of the defender's max HP."""
    try:
        if move.category == MoveCategory.STATUS:
            return 0.0
        if move.damage == "level":  # Seismic Toss / Night Shade
            if _type_multiplier(move.type, defender) == 0:
                return 0.0
            return (attacker.level or 80) / _max_hp(defender, defender_known)

        if tera and not attacker.is_terastallized:
            tera_type = _own_tera_type(attacker)
            original = attacker.original_types
            if tera_type is not None and move.type == tera_type:
                stab = 2.0 if tera_type in original else 1.5
            elif move.type in original:
                stab = 1.5
            else:
                stab = 1.0
        else:
            stab = attacker.stab_multiplier if move.type in attacker.types else 1.0

        return _damage_fraction(
            move.base_power,
            move.type,
            move.category == MoveCategory.PHYSICAL,
            attacker,
            defender,
            attacker_known,
            defender_known,
            stab,
            hits=move.expected_hits,
            accuracy=move.accuracy,
            defend_physical=move.defensive_category == MoveCategory.PHYSICAL,
        )
    except Exception:
        return 0.0


def _remaining_fraction(damage_fraction: float, defender: Pokemon) -> float:
    """Damage relative to the defender's *remaining* HP, capped at 1 (= KO)."""
    hp = max(defender.current_hp_fraction, 1e-3)
    return float(min(damage_fraction / hp, 1.0))


def _opponent_threat(opponent: Pokemon, target: Pokemon) -> float:
    """Predicts the damage the max-base-power opponent will do to `target`.

    The max agent always clicks its highest base power move, so if moves have
    been revealed we use the strongest one. Until all four moves are known we
    also consider a 90 BP STAB attack of each of its types (pessimistic).
    """
    threat = 0.0
    revealed = [m for m in opponent.moves.values() if m.base_power > 0]
    if revealed:
        predicted = max(revealed, key=lambda m: m.base_power)
        threat = _move_damage(predicted, opponent, target, False, True)
    if len(opponent.moves) < 4:
        physical = opponent.base_stats["atk"] >= opponent.base_stats["spa"]
        for mon_type in opponent.types:
            threat = max(
                threat,
                _damage_fraction(90, mon_type, physical, opponent, target, False, True, 1.5),
            )
    return threat


def _outspeeds(ours: Pokemon, theirs: Pokemon) -> float:
    our_speed = _stat(ours, "spe", True)
    their_speed = _stat(theirs, "spe", False)
    if our_speed == their_speed:
        return 0.5
    return 1.0 if our_speed > their_speed else 0.0


def _best_move_damage(mon: Pokemon, opponent: Pokemon) -> float:
    return max(
        (_move_damage(m, mon, opponent, True, False) for m in mon.moves.values()),
        default=0.0,
    )


def _switch_score(candidate: Pokemon, opponent: Pokemon) -> Tuple[float, float, float]:
    offence = _remaining_fraction(_best_move_damage(candidate, opponent), opponent)
    threat = _remaining_fraction(_opponent_threat(opponent, candidate), candidate)
    score = offence - threat + 0.25 * _outspeeds(candidate, opponent)
    return score, offence, threat


def _best_switch(battle: AbstractBattle) -> Tuple[Optional[Pokemon], float, float]:
    """Bench Pokemon with the best offence-minus-threat matchup."""
    opponent = battle.opponent_active_pokemon
    best, best_score, best_off, best_threat = None, -np.inf, 0.0, 1.0
    for candidate in battle.available_switches:
        if opponent is None:
            score, off, threat = candidate.current_hp_fraction, 0.0, 0.0
        else:
            score, off, threat = _switch_score(candidate, opponent)
        if score > best_score:
            best, best_score, best_off, best_threat = candidate, score, off, threat
    return best, best_off, best_threat


def _team_index(battle: AbstractBattle, mon: Pokemon) -> int:
    for i, team_mon in enumerate(battle.team.values()):
        if team_mon.base_species == mon.base_species:
            return i
    return SD_DEFAULT


# ===========================================================================
# Environment
# ===========================================================================
class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one
        self._episode_stats = self._new_episode_stats()

    # -----------------------------------------------------------------------
    # Episode bookkeeping (feeds get_additional_info for evaluation)
    # -----------------------------------------------------------------------
    @staticmethod
    def _new_episode_stats() -> Dict[str, float]:
        return {
            "decisions": 0,
            "move_actions": 0,
            "tera_actions": 0,
            "switch_actions": 0,
            "forced_switches": 0,
            "remapped_actions": 0,
            "greedy_agreement": 0,
            "super_effective": 0,
            "resisted": 0,
        }

    def reset(self, seed=None, options=None):
        self._episode_stats = self._new_episode_stats()
        return super().reset(seed, options)

    # -----------------------------------------------------------------------
    # Action space
    # -----------------------------------------------------------------------
    def _get_action_size(self) -> int | None:
        return ACTION_SIZE

    def process_action(self, action: np.int64) -> np.int64:
        """
        Maps the agent's high-level action onto the 26 Showdown actions.

        0 <= action <= 5: switch
        6 <= action <= 9: move
        22 <= action <= 25: move and terastallize
        """
        battle = self.battle1
        if battle is None or battle.finished or not getattr(self, "agent1_to_move", True):
            return np.int64(action)

        stats = self._episode_stats
        stats["decisions"] += 1
        action = int(action)

        # Forced switch (our active fainted / U-turn etc.) - pick best matchup
        if battle.force_switch or battle.active_pokemon is None:
            stats["forced_switches"] += 1
            best, _, _ = _best_switch(battle)
            return np.int64(SD_DEFAULT if best is None else _team_index(battle, best))

        # Struggle / recharge turns only have one legal option
        if len(battle.available_moves) == 1 and battle.available_moves[0].id in (
            "struggle",
            "recharge",
        ):
            return np.int64(SD_MOVE_OFFSET)

        if action == ACTION_SWITCH:
            best, _, _ = _best_switch(battle)
            if best is not None and not battle.trapped:
                stats["switch_actions"] += 1
                return np.int64(_team_index(battle, best))
            stats["remapped_actions"] += 1

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        slots = list(active.moves.values())
        legal_ids = {m.id for m in battle.available_moves}
        legal_slots = [
            i for i, m in enumerate(slots[:N_MOVE_SLOTS]) if m.id in legal_ids
        ]
        if not legal_slots:
            stats["remapped_actions"] += 1
            return np.int64(SD_DEFAULT)

        def damage(slot: int) -> float:
            if opponent is None:
                return slots[slot].base_power
            return _move_damage(slots[slot], active, opponent, True, False)

        best_slot = max(legal_slots, key=damage)

        slot = action % N_MOVE_SLOTS
        terastallize = ACTION_TERA_OFFSET <= action < ACTION_SWITCH
        if action == ACTION_SWITCH:
            slot = best_slot
        elif slot not in legal_slots:
            stats["remapped_actions"] += 1
            slot = best_slot

        stats["move_actions"] += 1
        if damage(slot) >= damage(best_slot) - 1e-6:
            stats["greedy_agreement"] += 1
        if opponent is not None:
            multiplier = _type_multiplier(slots[slot].type, opponent)
            if slots[slot].category != MoveCategory.STATUS:
                stats["super_effective"] += multiplier > 1
                stats["resisted"] += multiplier < 1

        if terastallize and battle.can_tera:
            stats["tera_actions"] += 1
            return np.int64(SD_TERA_OFFSET + slot)
        return np.int64(SD_MOVE_OFFSET + slot)

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------
    @staticmethod
    def _potential(battle: AbstractBattle) -> float:
        """Phi(s): material advantage = HP, faints and status of both teams.
        Unrevealed opponent Pokemon count as healthy."""
        our_hp = sum(m.current_hp_fraction for m in battle.team.values())
        our_fainted = sum(m.fainted for m in battle.team.values())
        our_status = sum(
            1 for m in battle.team.values() if m.status is not None and not m.fainted
        )

        opp = list(battle.opponent_team.values())
        unseen = max(TEAM_SIZE - len(opp), 0)
        opp_hp = sum(m.current_hp_fraction for m in opp) + unseen
        opp_fainted = sum(m.fainted for m in opp)
        opp_status = sum(1 for m in opp if m.status is not None and not m.fainted)

        return (
            HP_WEIGHT * (our_hp - opp_hp)
            + FAINT_WEIGHT * (opp_fainted - our_fainted)
            + STATUS_WEIGHT * (opp_status - our_status)
        )

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Potential-based shaping (Ng et al., 1999): r = Phi(s') - Phi(s), plus a
        terminal win/loss reward. The dense term gives feedback every turn
        while the terminal term keeps the objective aligned with winning.
        """
        prior_battle = self._get_prior_battle(battle)

        reward = self._potential(battle)
        if prior_battle is not None:
            reward -= self._potential(prior_battle)

        if battle.finished:
            if battle.won:
                reward += VICTORY_REWARD
            elif battle.lost:
                reward -= VICTORY_REWARD

        return float(reward)

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------
    def _observation_size(self) -> int:
        return OBS_SIZE

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        52 features, all in [0, 1]:
          [0:28]  per move slot: legal, expected dmg / opp remaining HP, type
                  multiplier / 4, is status, has priority, accuracy, expected
                  dmg if we terastallize
          [28:40] active matchup: our HP, opp HP, we outspeed, predicted opp
                  dmg / our remaining HP, our status, opp status, our
                  atk/spa/spe boosts, opp atk/spa/spe boosts
          [40:44] best switch-in: switch possible, its offence, the threat it
                  faces, its HP
          [44:48] team: our alive, opp alive, our total HP, opp total HP
          [48:52] flags: can tera, forced switch, trapped, turn / 50
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # --- per-move features --------------------------------------------
        if active is not None and opponent is not None:
            legal_ids = {m.id for m in battle.available_moves}
            for i, move in enumerate(list(active.moves.values())[:N_MOVE_SLOTS]):
                base = i * MOVE_FEATURES
                dmg = _move_damage(move, active, opponent, True, False)
                tera_dmg = (
                    _move_damage(move, active, opponent, True, False, tera=True)
                    if battle.can_tera
                    else dmg
                )
                obs[base + 0] = float(move.id in legal_ids)
                obs[base + 1] = _remaining_fraction(dmg, opponent)
                obs[base + 2] = _type_multiplier(move.type, opponent) / 4.0
                obs[base + 3] = float(move.category == MoveCategory.STATUS)
                obs[base + 4] = float(move.priority > 0)
                obs[base + 5] = move.accuracy
                obs[base + 6] = _remaining_fraction(tera_dmg, opponent)

        # --- active matchup -----------------------------------------------
        i = N_MOVE_SLOTS * MOVE_FEATURES
        if active is not None:
            obs[i + 0] = active.current_hp_fraction
            obs[i + 4] = float(active.status is not None)
            for k, stat in enumerate(BOOST_STATS):
                obs[i + 6 + k] = (active.boosts.get(stat, 0) + 6) / 12
        if opponent is not None:
            obs[i + 1] = opponent.current_hp_fraction
            obs[i + 5] = float(opponent.status is not None)
            for k, stat in enumerate(BOOST_STATS):
                obs[i + 9 + k] = (opponent.boosts.get(stat, 0) + 6) / 12
        if active is not None and opponent is not None:
            obs[i + 2] = _outspeeds(active, opponent)
            obs[i + 3] = _remaining_fraction(_opponent_threat(opponent, active), active)

        # --- best switch-in -----------------------------------------------
        i += 12
        best, best_off, best_threat = _best_switch(battle)
        if best is not None and not battle.trapped:
            obs[i + 0] = 1.0
            obs[i + 1] = best_off
            obs[i + 2] = best_threat
            obs[i + 3] = best.current_hp_fraction

        # --- team totals ----------------------------------------------------
        i += 4
        team = list(battle.team.values())
        opp_team = list(battle.opponent_team.values())
        unseen = max(TEAM_SIZE - len(opp_team), 0)
        obs[i + 0] = sum(not m.fainted for m in team) / TEAM_SIZE
        obs[i + 1] = (sum(not m.fainted for m in opp_team) + unseen) / TEAM_SIZE
        obs[i + 2] = sum(m.current_hp_fraction for m in team) / TEAM_SIZE
        obs[i + 3] = (sum(m.current_hp_fraction for m in opp_team) + unseen) / TEAM_SIZE

        # --- flags --------------------------------------------------------
        i += 4
        obs[i + 0] = float(battle.can_tera)
        obs[i + 1] = float(battle.force_switch)
        obs[i + 2] = float(battle.trapped)
        obs[i + 3] = min(battle.turn / 50.0, 1.0)

        return np.clip(np.nan_to_num(obs), 0.0, 1.0).astype(np.float32)

    # -----------------------------------------------------------------------
    # Logging - per-episode metrics for the report
    # -----------------------------------------------------------------------
    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        battle = self.battle1
        if battle is not None:
            agent = self.possible_agents[0]
            stats = self._episode_stats
            moves = max(stats["move_actions"], 1)
            decisions = max(stats["decisions"], 1)
            opp_team = list(battle.opponent_team.values())
            unseen = max(TEAM_SIZE - len(opp_team), 0)

            info[agent]["win"] = battle.won
            info[agent]["turns"] = battle.turn
            info[agent]["remaining_mons"] = sum(not m.fainted for m in battle.team.values())
            info[agent]["opp_remaining_mons"] = (
                sum(not m.fainted for m in opp_team) + unseen
            )
            info[agent]["hp_advantage"] = float(
                sum(m.current_hp_fraction for m in battle.team.values())
                - sum(m.current_hp_fraction for m in opp_team)
                - unseen
            )
            info[agent]["greedy_move_rate"] = stats["greedy_agreement"] / moves
            info[agent]["super_effective_rate"] = stats["super_effective"] / moves
            info[agent]["resisted_rate"] = stats["resisted"] / moves
            info[agent]["switch_rate"] = stats["switch_actions"] / decisions
            info[agent]["tera_used"] = stats["tera_actions"]
            info[agent]["remapped_rate"] = stats["remapped_actions"] / decisions

        return info


########################################
# DO NOT EDIT BELOW THIS LINE
########################################

class SingleShowdownWrapper(SingleAgentWrapper):
    """
    Wrapper for single-agent training against specified opponents.
    """

    def __init__(self, team_type: str = "random", opponent_type: str = "random", evaluation: bool = False):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"
        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)
        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")
        bot_teams = {}
        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8") as file:
                    bot_teams[team_file[:-4]] = file.read()
        return bot_teams.get(team_type, None)
