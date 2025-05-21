import math
from z3 import Solver, BitVec, BitVecVal, URem, ZeroExt, Extract, sat, unsat, BitVecNumRef, Z3Exception
import time

# --- LCG parameters and derived constants ---
A_val = 68909602460261
W = 48  # Bit-width
MODULUS_LCG = 2**W  # 2^48
MOD47 = 47
RX = 42  # Target remainder for X_j (X_j % 47 == 42)
RQ = 46  # Target remainder for Q_j (Q_j % 47 == 46)

# For Z3 BitVec operations
A_BV_W = BitVecVal(A_val, W)
A_BV_96 = BitVecVal(A_val, 96)
MOD47_BV_W = BitVecVal(MOD47, W)
RX_BV_W = BitVecVal(RX, W)
RQ_BV_W = BitVecVal(RQ, W)

inv47_mod_2_48 = pow(MOD47, -1, MODULUS_LCG)
C_prime = ((A_val - 1) * RX * inv47_mod_2_48) % MODULUS_LCG
C_prime_BV_W = BitVecVal(C_prime, W)
C_prime_BV_96 = BitVecVal(C_prime, 96)


def find_globally_optimal_k_and_y0_smt_min_model(k_search_limit=15):
    """
    Uses SMT incrementally to find the highest k and a Y0 that achieves it.
    Minimizes model extraction during the k-search loop.
    k_search_limit: Max k to check. The true k_max for this problem is small (5).
    """
    solver = Solver()
    Y0_sym = BitVec('Y0_smt', W)
    # Add a trivial constraint that involves Y0_sym to ensure Z3 considers it.
    solver.add(Y0_sym == Y0_sym)

    current_Y_sym = Y0_sym
    max_k_achieved = -1

    # Check k=0 (X0 implicitly valid via Y0 transform, no Q condition yet)
    if solver.check() == sat:
        max_k_achieved = 0
        print(f"SMT Search: k=0 is SAT.")
        solver.push()
    else:
        print("SMT Search: k=0 is UNSAT (unexpected).")
        return -1, None

    # Attempt to extend the streak for k = 1, 2, ... up to k_search_limit
    for k_being_tested in range(1, k_search_limit + 1):
        print(
            f"SMT Search: Attempting to satisfy conditions for k={k_being_tested}...")

        Y_next_sym = BitVec(f'Y_smt_{k_being_tested}', W)

        # 1. Y-Recurrence: Y_{k} = (A * Y_{k-1} + C_prime) mod M
        y_recurrence_prod = ZeroExt(W, current_Y_sym) * A_BV_96 + C_prime_BV_96
        solver.add(Y_next_sym == Extract(W-1, 0, y_recurrence_prod))

        # 2. Quotient Condition for Q_{k-1}:
        #    X_{k-1} = 47 * Y_{k-1} (current_Y_sym) + RX
        #    Q_{k-1} = floor(A * X_{k-1} / M)
        #    Q_{k-1} % 47 == RQ
        X_prev_symbolic = MOD47_BV_W * current_Y_sym + RX_BV_W
        product_for_Q_prev = ZeroExt(W, X_prev_symbolic) * A_BV_96
        Q_prev_sym = Extract(95, 48, product_for_Q_prev)  # This is Q_{k-1}
        solver.add(URem(Q_prev_sym, MOD47_BV_W) == RQ_BV_W)

        if solver.check() == sat:
            max_k_achieved = k_being_tested
            current_Y_sym = Y_next_sym  # Update current_Y_sym for the next iteration
            print(f"SMT Search: k={max_k_achieved} is SAT.")
            solver.push()  # Save this new SAT state
        else:
            print(
                f"SMT Search: k={k_being_tested} is UNSAT. Previous k={max_k_achieved} is the global maximum.")
            solver.pop()  # Revert to the last valid SAT state
            break  # Stop searching, max_k_achieved is the global max

    y0_final_value = None
    if max_k_achieved >= 0:
        # Solver is now in the state of the last successful push (corresponding to max_k_achieved)
        # Re-check SAT and extract the model only once at the end.
        if solver.check() == sat:
            model = solver.model()
            y0_eval = model.eval(Y0_sym, model_completion=True)
            if isinstance(y0_eval, BitVecNumRef):
                y0_final_value = y0_eval.as_long()
                print(
                    f"SMT Search: Final Y0 for k={max_k_achieved} is {y0_final_value}.")
            else:
                print(
                    f"SMT Search: Final Y0 eval for k={max_k_achieved} was not a concrete number.")
        else:
            print(
                f"SMT Search: Warning! Solver state for final k={max_k_achieved} became UNSAT (should be SAT).")

    return max_k_achieved, y0_final_value

# --- Verification Function (using original X_i, assumed correct from previous discussions) ---


def verify_smt_all_conditions(x0_original_int, k_to_verify_int):
    # ... (This function can be copied from the previous response, it's for verification) ...
    # ... (For brevity, I'll skip pasting it again, but it's needed) ...
    A_bv_96 = BitVecVal(A_val, 96)
    MOD47_bv_W = BitVecVal(MOD47, W)
    RX_bv_W = BitVecVal(RX, W)
    RQ_bv_W = BitVecVal(RQ, W)
    solver = Solver()
    num_states_needed = k_to_verify_int + 2
    X_sym = [BitVec(f"X_smt_verify_{i}", W) for i in range(num_states_needed)]
    solver.add(X_sym[0] == BitVecVal(x0_original_int, W))

    prefix_is_valid = True
    solver.push()
    for i in range(k_to_verify_int + 1):
        solver.add(URem(X_sym[i], MOD47_bv_W) == RX_bv_W)
        prod_X = ZeroExt(W, X_sym[i]) * A_bv_96
        Q_i_sym = Extract(95, 48, prod_X)
        X_next_sym_val = Extract(47, 0, prod_X)
        solver.add(X_sym[i + 1] == X_next_sym_val)
        if i < k_to_verify_int:
            solver.add(URem(Q_i_sym, MOD47_bv_W) == RQ_bv_W)
    if solver.check() != sat:
        prefix_is_valid = False
    solver.pop()

    print("\n🔎 SMT Verification Details:")
    print(f"  - Prefix X₀..X_{k_to_verify_int} valid (all X_j and relevant Q_j conditions met)? ",
          "✅ yes" if prefix_is_valid else "❌ no")
    if not prefix_is_valid:
        return False

    solver.push()
    for i in range(k_to_verify_int + 1):
        solver.add(URem(X_sym[i], MOD47_bv_W) == RX_bv_W)
        prod_X_check = ZeroExt(W, X_sym[i]) * A_bv_96
        Q_i_check_sym = Extract(95, 48, prod_X_check)
        X_next_check_sym_val = Extract(47, 0, prod_X_check)
        solver.add(X_sym[i+1] == X_next_check_sym_val)
        if i < k_to_verify_int:
            solver.add(URem(Q_i_check_sym, MOD47_bv_W) == RQ_bv_W)

    prod_X_k_verify = ZeroExt(W, X_sym[k_to_verify_int]) * A_bv_96
    Q_k_verify_sym = Extract(95, 48, prod_X_k_verify)
    solver.add(URem(Q_k_verify_sym, MOD47_bv_W) == RQ_bv_W)
    solver.add(URem(X_sym[k_to_verify_int + 1], MOD47_bv_W) == RX_bv_W)
    streak_extends_further = (solver.check() == sat)
    solver.pop()
    print(f"  - Streak extends to k={k_to_verify_int + 1} (all X_j and Q_j conditions met)? ",
          "⚠️ yes (...)" if streak_extends_further else "✅ no (...)")
    return prefix_is_valid and not streak_extends_further


# --- Main Execution ---
if __name__ == "__main__":
    t_start_total = time.time()

    print("--- Starting Pure SMT Search for Optimal k and X0 (minimized model extraction) ---")
    # For this problem, k_max is known to be 5. So limit should be > 5.
    # Setting to 7 for a margin. If it takes hours for k=4, this will also be slow.
    k_smt_found, y0_smt_found = find_globally_optimal_k_and_y0_smt_min_model(
        k_search_limit=7)
    t_smt_search_end = time.time()

    if y0_smt_found is not None and k_smt_found != -1:
        x0_smt_val = RX + MOD47 * y0_smt_found

        print(f"\n--- SMT Search Result ---")
        print(f"  Max k found by SMT search   : {k_smt_found}")
        print(f"  Corresponding Y0 from SMT   : {y0_smt_found}")
        print(f"  Corresponding X0 from SMT   : {x0_smt_val}")
        print(
            f"  SMT search time             : {(t_smt_search_end - t_start_total):.2f} s")

        print(
            f"\nStarting SMT verification for X0 = {x0_smt_val}, k = {k_smt_found}...")
        is_verified_by_smt = verify_smt_all_conditions(x0_smt_val, k_smt_found)
        t_smt_verify_end = time.time()

        print(f"\n✅ Final SMT-Based Result:")
        print(f"  Max k (globally optimal)    = {k_smt_found}")
        print(f"  An optimal X0               = {x0_smt_val}")
        print(f"  SMT Verified as maximal     = {is_verified_by_smt}")
        print(
            f"  SMT verification time       = {(t_smt_verify_end - t_smt_search_end):.2f} s")
    else:
        print("❌ No solution found by the SMT search.")

    print(f"⏱️ Total execution time = {(time.time() - t_start_total):.2f} s")
