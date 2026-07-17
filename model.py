"""
Market-Making & Betting-Game Simulator

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - expected_value
import numpy as np

def expected_value(values, probabilities):
    """
    Computes the expected value of a discrete distribution.
    
    Args:
        values: 1D array-like of outcome values (v_i).
        probabilities: 1D array-like of corresponding probabilities (p_i).
        
    Returns:
        float: The expected value E[X] as a native Python float.
    """
    # Convert both inputs to numpy arrays to uniformly handle lists, tuples, or arrays
    v = np.asarray(values, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    
    # Compute the weighted sum: E[X] = \sum (v_i * p_i)
    weighted_sum = np.dot(v, p)
    
    return float(weighted_sum)

# Step 2 - one_reroll_die_value
def one_reroll_die_value(sides):
    """
    Computes the optimal keep-or-reroll strategy and expected winnings for an N-sided die.
    
    Args:
        sides (int): The number of faces on the die (1 to sides).
        
    Returns:
        dict: A dictionary containing 'value' (expected winnings) and 'reroll_faces' (list of integers).
    """
    # Create the die faces
    faces = list(range(1, sides + 1))
    probs = [1 / sides] * sides
    
    # 1. Calculate the baseline expected value of a single roll
    e_single_roll = expected_value(faces, probs)
    
    # 2. Identify the optimal policy: reroll any face worse than or equal to our reroll baseline
    reroll_faces = [face for face in faces if face < e_single_roll]
    
    # 3. Build the effective payouts for each possible first roll under optimal play
    optimal_payouts = [max(face, e_single_roll) for face in faces]
    
    # 4. Calculate the total expected winnings of the game
    e_game = expected_value(optimal_payouts, probs)
    
    return {
        'value': e_game,
        'reroll_faces': reroll_faces
    }

# Step 3 - pay_per_reroll_die_game
def pay_per_reroll_die_game(sides, reroll_cost):
    """
    Finds the optimal keep threshold and expected net winnings for an infinite reroll game with costs.
    
    Args:
        sides (int): The number of faces on the die (1 to sides).
        reroll_cost (float): The cost paid each time a reroll is chosen.
        
    Returns:
        dict: {'threshold': int, 'value': float} representing the optimal strategy and expected winnings.
    """
    best_threshold = 1
    best_value = float('-inf')
    
    # Test every candidate threshold from 1 up to 'sides'
    for threshold in range(1, sides + 1):
        # 1. Count how many faces cause us to keep vs. reroll
        num_keep = sides - threshold + 1
        num_reroll = threshold - 1
        
        # 2. Calculate the average gross payout of the faces we keep
        e_keep = (threshold + sides) / 2.0
        
        # 3. Calculate the expected number of rerolls before landing on a keep face
        expected_rerolls = num_reroll / num_keep
        
        # 4. Calculate net expected value for this threshold
        net_value = e_keep - (expected_rerolls * reroll_cost)
        
        # 5. Update best strategy
        if net_value > best_value:
            best_value = net_value
            best_threshold = threshold
            
    return {
        'threshold': best_threshold,
        'value': float(best_value)
    }

# Step 4 - red_black_card_game_value
def red_black_card_game_value(num_red, num_black):
    # TODO: return {'value': expected payout under optimal stopping, 'stop_now': whether to stop immediately}.
    pass
from functools import lru_cache

def red_black_card_game_value(num_red, num_black):
    """
    Computes the optimal expected payout and initial stop decision for the Red-Black card game.
    
    Args:
        num_red (int): Initial number of red cards (+1 value).
        num_black (int): Initial number of black cards (-1 value).
        
    Returns:
        dict: {'value': float, 'stop_now': bool}
    """
    
    @lru_cache(maxsize=None)
    def get_continuation_value(r, b):
        # Base Case 1: No cards left, or only Black cards left (-1 each). 
        # We would never draw a black card on purpose, so our future gain is 0.
        if r == 0:
            return 0.0
            
        # Base Case 2: Only Red cards left (+1 each). 
        # We draw all of them for a guaranteed win!
        if b == 0:
            return float(r)
            
        # Recursive Case: We have both red and black cards remaining.
        total_cards = r + b
        prob_red = r / total_cards
        prob_black = b / total_cards
        
        # If we draw Red (+1), we transition to state (r - 1, b)
        # If we draw Black (-1), we transition to state (r, b - 1)
        expected_draw_value = (
            prob_red * (1.0 + get_continuation_value(r - 1, b)) +
            prob_black * (-1.0 + get_continuation_value(r, b - 1))
        )
        
        # Optimal policy: We can stop at any time for 0 additional payout.
        # So the value of this state is the maximum between stopping (0.0) and drawing.
        return max(0.0, expected_draw_value)

    # Calculate the raw expected value of drawing from our starting deck WITHOUT taking max(0, ...) yet
    # Why? Because we need to know if the draw itself is <= 0 to determine if we should stop immediately!
    if num_red == 0:
        raw_draw_value = 0.0
    elif num_black == 0:
        raw_draw_value = float(num_red)
    else:
        total = num_red + num_black
        raw_draw_value = (
            (num_red / total) * (1.0 + get_continuation_value(num_red - 1, num_black)) +
            (num_black / total) * (-1.0 + get_continuation_value(num_red, num_black - 1))
        )
    
    # We stop immediately if drawing yields 0 or negative expected value
    stop_now = (raw_draw_value <= 0.0)
    
    # Our final game value is at least 0 (since we can just choose to stop immediately)
    game_value = max(0.0, raw_draw_value)
    
    return {
        'value': float(game_value),
        'stop_now': stop_now
    }

# Step 5 - make_quotes
def make_quotes(fair_value, spread_width):
    """
    Generates symmetric two-sided quotes around a fair value.
    """
    # Calculate the half-spread distance from the midpoint
    half_spread = spread_width / 2.0
    
    # Subtract half-spread for the buy price, add for the sell price
    return {
        'bid': float(fair_value - half_spread),
        'ask': float(fair_value + half_spread)
    }

# Step 6 - execute_trade
def execute_trade(state, side, bid, ask, size=1.0):
    """
    Applies a counterparty trade against quoted prices and returns a new updated state.
    """
    # Extract current balance and inventory without mutating original state
    cash = state['cash']
    inv = state['inventory']
    
    if side == 'buy':
        # Counterparty buys from you -> YOU sell at ask: cash up, inventory down
        return {'cash': cash + (size * ask), 'inventory': inv - size}
    elif side == 'sell':
        # Counterparty sells to you -> YOU buy at bid: cash down, inventory up
        return {'cash': cash - (size * bid), 'inventory': inv + size}
    else:
        # Fallback to unchanged state if an unknown trade side is passed
        return {'cash': cash, 'inventory': inv}

# Step 7 - mark_to_market_pnl
def mark_to_market_pnl(cash, inventory, settlement_value):
    """
    Computes total profit and loss by valuing remaining inventory at the final settlement price.
    """
    # Total PnL is cash balance plus the liquidation value of inventory
    return float(cash + (inventory * settlement_value))

# Step 8 - adverse_selection_loss
import numpy as np

def adverse_selection_loss(fair_value, bid, ask, informed_values, informed_probabilities):
    """
    Computes the expected loss to an informed counterparty who knows the true value.
    """
    # Convert input sequences to uniform NumPy float arrays
    v = np.asarray(informed_values, dtype=float)
    p = np.asarray(informed_probabilities, dtype=float)
    
    # Calculate non-negative losses when true value breaches either the bid or ask
    loss_per_val = np.maximum(v - ask, 0.0) + np.maximum(bid - v, 0.0)
    
    # Return the weighted average loss across all possible true values as a Python float
    return float(np.dot(loss_per_val, p))

# Step 9 - uncertainty_spread (not yet solved)
# TODO: implement

# Step 10 - inventory_skewed_quotes (not yet solved)
# TODO: implement

# Step 11 - update_fair_value_from_trade (not yet solved)
# TODO: implement

# Step 12 - update_remaining_card_value (not yet solved)
# TODO: implement

# Step 13 - run_market_making_episode (not yet solved)
# TODO: implement

# Step 14 - summarize_episode_pnls (not yet solved)
# TODO: implement

