from z3 import Solver, BitVec, BitVecVal, URem, ZeroExt, Extract, sat
import time

# LCG parameters
A_val = 68909602460261
WIDTH = 48
MODULUS_LCG = 2**WIDTH
MOD47_val = 47
TARGET_REM_val = 42
QUOTIENT_REM_val = 46

# --- Python Simulation ---


def compute_next_state_python(current_x):
    return (A_val * current_x) % MODULUS_LCG


def compute_quotient_python(current_x):
    return (A_val * current_x) // MODULUS_LCG


def simulate_k_for_seed_python(x0, max_k_to_simulate=10):
    """
    Simulates the LCG for a given x0 and returns the highest k achieved.
    k is the highest index j such that X_j satisfies the condition.
    So, k=0 means X0 is good, X1 is bad.
    Returns -1 if X0 itself is bad.
    """
    current_x = x0
    # Check X0 condition
    if current_x % MOD47_val != TARGET_REM_val:
        return -1  # X0 itself is bad

    # k_idx goes from 0 to max_k_to_simulate
    for k_idx in range(max_k_to_simulate + 1):
        # X_k_idx is current_x
        # We need to check X_k_idx and the quotient Q_k_idx that generates X_{k_idx+1}

        # For the last k_idx, we only check X_k_idx, not the quotient leading from it.
        # Successfully checked up to X_{max_k_to_simulate}
        if k_idx == max_k_to_simulate:
            if current_x % MOD47_val == TARGET_REM_val:
                return k_idx
            else:  # Should not happen if previous step was fine
                return k_idx - 1

        # Check X_k_idx (already done for k_idx=0)
        if k_idx > 0 and (current_x % MOD47_val != TARGET_REM_val):
            return k_idx - 1  # Streak broken at X_k_idx

        # Check quotient Q_k_idx
        quotient = compute_quotient_python(current_x)
        if quotient % MOD47_val != QUOTIENT_REM_val:
            # Streak broken by Q_k_idx (so X_k_idx was the last good state)
            return k_idx

        current_x = compute_next_state_python(
            current_x)  # This is now X_{k_idx+1}

    # Should be covered by the loop logic, but as a fallback
    return max_k_to_simulate


def find_best_seed_via_simulation(num_candidates_to_sample=5*10**6, max_k_simulation=10):
    """
    Searches for the best seed X0 by simulating a number of candidates.
    """
    best_k_found = -1
    best_seed_found = None

    # Iterate through X0 candidates: X0 % 47 == 42
    # Start from TARGET_REM_val, step by MOD47_val
    current_x0_candidate = TARGET_REM_val
    candidates_checked = 0

    print(
        f"Starting Python simulation, sampling up to {num_candidates_to_sample} candidates...")
    while current_x0_candidate < MODULUS_LCG and candidates_checked < num_candidates_to_sample:
        if candidates_checked % 100000 == 0 and candidates_checked > 0:
            print(
                f"  ...simulated {candidates_checked} candidates. Best k so far: {best_k_found} for X0={best_seed_found}")

        k_for_this_seed = simulate_k_for_seed_python(
            current_x0_candidate, max_k_simulation)

        if k_for_this_seed > best_k_found:
            best_k_found = k_for_this_seed
            best_seed_found = current_x0_candidate
            print(
                f"  New best: k={best_k_found}, X0={best_seed_found} (at candidate count {candidates_checked})")

        current_x0_candidate += MOD47_val
        candidates_checked += 1

    print(
        f"Python simulation finished. Checked {candidates_checked} candidates.")
    return best_k_found, best_seed_found

# --- Z3 Verification ---


def verify_with_z3(x0_to_verify, k_to_verify):
    """
    Verifies with Z3 that x0_to_verify achieves exactly k_to_verify.
    - Checks if k_to_verify is possible.
    - Checks if k_to_verify + 1 is NOT possible.
    Returns (True, True) if k is possible and k+1 is not.
             (True, False) if k is possible and k+1 is also possible (k was not maximal for this seed)
             (False, _) if k is not possible.
    """
    if x0_to_verify is None or k_to_verify < 0:
        print("Z3: Invalid input for verification.")
        return False, False

    print(f"Z3: Verifying if X0={x0_to_verify} achieves k={k_to_verify}...")

    s = Solver()
    A_bv = BitVecVal(A_val, WIDTH * 2)  # For 96-bit product
    A_bv_short = BitVecVal(A_val, WIDTH)  # For 48-bit recurrence

    # Symbolic states X_0, ..., X_{k_to_verify+1}
    # We need k_to_verify + 1 states for checking k_to_verify (X0...Xk)
    # And k_to_verify + 2 states for checking k_to_verify + 1 (X0...Xk+1)
    num_states_for_k_plus_1 = k_to_verify + 2
    X = [BitVec(f"X{i}", WIDTH) for i in range(num_states_for_k_plus_1)]

    # --- Check if k_to_verify is possible ---
    s.push()
    s.add(X[0] == x0_to_verify)

    # Constraints for X_0, ..., X_{k_to_verify}
    # This means k_to_verify+1 states, and k_to_verify quotients
    for j in range(k_to_verify + 1):  # j from 0 to k_to_verify
        s.add(URem(X[j], MOD47_val) == TARGET_REM_val)

        if j < k_to_verify:  # For Q_0, ..., Q_{k_to_verify-1}
            prod = ZeroExt(WIDTH, X[j]) * A_bv
            quotient = Extract(WIDTH*2 - 1, WIDTH, prod)  # High part
            next_state = Extract(WIDTH - 1, 0, prod)     # Low part

            s.add(URem(quotient, MOD47_val) == QUOTIENT_REM_val)
            s.add(X[j+1] == next_state)  # Z3 will ensure this matches LCG
            # s.add(X[j+1] == (X[j] * A_bv_short)) # Explicit LCG, also fine

    print(
        f"Z3: Checking satisfiability for k={k_to_verify} with X0={x0_to_verify}...")
    k_is_sat = (s.check() == sat)
    s.pop()  # Roll back constraints for k

    if not k_is_sat:
        print(f"Z3: k={k_to_verify} is UNSAT for X0={x0_to_verify}.")
        return False, False
    else:
        print(f"Z3: k={k_to_verify} is SAT for X0={x0_to_verify}.")

    # --- Check if k_to_verify + 1 is possible (i.e., if k_to_verify was NOT maximal) ---
    s.push()
    s.add(X[0] == x0_to_verify)

    # Constraints for X_0, ..., X_{k_to_verify+1}
    # This means k_to_verify+2 states, and k_to_verify+1 quotients
    target_k_plus_1 = k_to_verify + 1
    for j in range(target_k_plus_1 + 1):  # j from 0 to k_to_verify+1
        s.add(URem(X[j], MOD47_val) == TARGET_REM_val)

        if j < target_k_plus_1:  # For Q_0, ..., Q_{k_to_verify}
            prod = ZeroExt(WIDTH, X[j]) * A_bv
            quotient = Extract(WIDTH*2 - 1, WIDTH, prod)
            next_state = Extract(WIDTH - 1, 0, prod)

            s.add(URem(quotient, MOD47_val) == QUOTIENT_REM_val)
            s.add(X[j+1] == next_state)
            # s.add(X[j+1] == (X[j] * A_bv_short))

    print(
        f"Z3: Checking satisfiability for k={target_k_plus_1} with X0={x0_to_verify}...")
    k_plus_1_is_sat = (s.check() == sat)
    s.pop()

    if k_plus_1_is_sat:
        print(
            f"Z3: k={target_k_plus_1} is SAT for X0={x0_to_verify} (meaning k={k_to_verify} was not maximal for this seed).")
    else:
        print(
            f"Z3: k={target_k_plus_1} is UNSAT for X0={x0_to_verify} (meaning k={k_to_verify} was maximal for this seed).")

    # (k was possible, k+1 was not possible)
    return k_is_sat, not k_plus_1_is_sat


# --- Main Execution ---
if __name__ == "__main__":
    start_time_sim = time.time()
    # The known seed 215565119 is the 4,586,491st candidate if we start from 42 and step by 47.
    # So, sample_size should be around 5 million to find it.
    # Set max_k_simulation higher than expected k to not cut it short.
    # Max k = 4 for this problem, so we need 5 states X0...X4.
    # Thus, simulate_k_for_seed_python should return 4.

    # For a quicker test, reduce num_candidates_to_sample, but you might not find the optimal.
    # num_candidates_to_sample = 100_000 # Faster, but might not find k=4
    # Should be enough to find k=4 with X0=215565119
    num_candidates_to_sample = 5_000_0000000

    k_sim, seed_sim = find_best_seed_via_simulation(
        num_candidates_to_sample=num_candidates_to_sample,
        max_k_simulation=6  # Max index we expect to satisfy
    )
    end_time_sim = time.time()
    print(f"\nPython Simulation Result:")
    print(f"  Best k found: {k_sim}")
    print(f"  Seed X0: {seed_sim}")
    print(f"  Simulation time: {end_time_sim - start_time_sim:.2f} seconds")

    if seed_sim is not None and k_sim >= 0:
        start_time_z3 = time.time()
        k_possible, k_is_maximal_for_seed = verify_with_z3(seed_sim, k_sim)
        end_time_z3 = time.time()
        print(f"\nZ3 Verification Result for X0={seed_sim}, k={k_sim}:")
        if k_possible and k_is_maximal_for_seed:
            print(
                f"  SUCCESS: Z3 confirmed X0={seed_sim} achieves maximal k={k_sim}.")
        elif k_possible and not k_is_maximal_for_seed:
            print(
                f"  INFO: Z3 confirmed X0={seed_sim} achieves k={k_sim}, but k+1 is also possible for this seed.")
            print(f"        (This suggests the Python simulation might have stopped early for this seed or k_sim is not the true max for this seed).")
        else:  # not k_possible
            print(
                f"  FAILURE: Z3 could NOT confirm X0={seed_sim} achieves k={k_sim}.")
        print(
            f"  Z3 verification time: {end_time_z3 - start_time_z3:.2f} seconds")
    else:
        print("\nNo suitable seed found by Python simulation to verify with Z3.")
