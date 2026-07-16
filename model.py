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

# Step 4 - red_black_card_game_value (not yet solved)
# TODO: implement

# Step 5 - make_quotes (not yet solved)
# TODO: implement

# Step 6 - execute_trade (not yet solved)
# TODO: implement

# Step 7 - mark_to_market_pnl (not yet solved)
# TODO: implement

# Step 8 - adverse_selection_loss (not yet solved)
# TODO: implement

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

