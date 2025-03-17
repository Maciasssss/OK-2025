from ortools.linear_solver import pywraplp
import sys
import time
import gc
import math
import os


def solve_for_specific_k(alphabet, S, T, K, time_limit=60000):
    """
    Solves the MIFSA problem for a specific K value using linear programming.
    Returns (is_possible, transitions, accepting_states)
    """
    print(f"Attempting to solve for K={K}...")
    start_time = time.time()

    problem_size = K * len(alphabet) * len(S + T) * max(len(s) for s in S + T)
    if problem_size > 1000000:
        time_limit = min(time_limit, 30000)

    time_limit = int(time_limit)

    solver = None
    try:
        solver = pywraplp.Solver.CreateSolver('SCIP')
    except:
        try:
            solver = pywraplp.Solver.CreateSolver('CBC')
        except:
            print("Error: Neither SCIP nor CBC solver is available")
            return False, None, None

    if not solver:
        return False, None, None

    solver.SetTimeLimit(time_limit)
    solver.SetNumThreads(4)

    try:
        solver.SetLogLevel(0)
    except:
        pass

    if K > 5:
        try:
            solver.SetPresolveLevel(0)
        except:
            pass

    try:
        accepting = {}
        transition = {}

        for q in range(K):
            accepting[q] = solver.BoolVar(f'acc_{q}')

        use_compact_naming = problem_size > 500000
        for q in range(K):
            for a in alphabet:
                for q_prime in range(K):
                    if use_compact_naming:
                        a_idx = alphabet.index(a)
                        var_name = f'd_{q*len(alphabet)*K + a_idx*K + q_prime}'
                    else:
                        var_name = f'd_{q}_{a}_{q_prime}'
                    transition[(q, a, q_prime)] = solver.BoolVar(var_name)

        for q in range(K):
            for a in alphabet:
                solver.Add(sum(transition[(q, a, q_prime)]
                           for q_prime in range(K)) == 1)

        string_length_groups = {}
        for idx, s in enumerate(S):
            length = len(s)
            if length not in string_length_groups:
                string_length_groups[length] = []
            string_length_groups[length].append((True, s, idx))

        for idx, t in enumerate(T):
            length = len(t)
            if length not in string_length_groups:
                string_length_groups[length] = []
            string_length_groups[length].append((False, t, idx))

        string_count = 0

        for length, string_group in string_length_groups.items():
            for is_accept, string, _ in string_group:
                string_count += 1

                state = {}
                for q in range(K):
                    state[(0, q)] = solver.BoolVar(f's_{string_count}_{0}_{q}')
                    solver.Add(state[(0, q)] == (1 if q == 0 else 0))

                for i in range(length):
                    a = string[i]

                    for q in range(K):
                        state[(i+1, q)
                              ] = solver.BoolVar(f's_{string_count}_{i+1}_{q}')

                    for q_next in range(K):
                        incoming_sum = []

                        for q_current in range(K):
                            path_var = solver.BoolVar(
                                f'p_{string_count}_{i}_{q_current}_{q_next}')
                            solver.Add(path_var <= state[(i, q_current)])
                            solver.Add(
                                path_var <= transition[(q_current, a, q_next)])
                            solver.Add(
                                path_var >= state[(i, q_current)] + transition[(q_current, a, q_next)] - 1)
                            incoming_sum.append(path_var)

                        solver.Add(state[(i+1, q_next)] == sum(incoming_sum))

                if is_accept:
                    accept_expr = []
                    for q in range(K):
                        accept_var = solver.BoolVar(f'a_{string_count}_{q}')
                        solver.Add(accept_var <= state[(length, q)])
                        solver.Add(accept_var <= accepting[q])
                        solver.Add(accept_var >= state[(
                            length, q)] + accepting[q] - 1)
                        accept_expr.append(accept_var)

                    solver.Add(sum(accept_expr) >= 1)
                else:
                    reject_expr = []
                    for q in range(K):
                        reject_var = solver.BoolVar(f'r_{string_count}_{q}')
                        solver.Add(reject_var <= state[(length, q)])
                        solver.Add(reject_var <= 1 - accepting[q])
                        solver.Add(
                            reject_var >= state[(length, q)] - accepting[q])
                        reject_expr.append(reject_var)

                    solver.Add(sum(reject_expr) >= 1)

            if problem_size > 500000:
                gc.collect()

        print(f"Starting solver...")
        solver_start = time.time()
        status = solver.Solve()
        solver_time = time.time() - solver_start
        print(f"Solver finished in {solver_time:.2f}s")

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            dfa_transitions = {}
            for q in range(K):
                for a in alphabet:
                    for q_prime in range(K):
                        if transition[(q, a, q_prime)].solution_value() > 0.5:
                            dfa_transitions[(q, a)] = q_prime
                            break

            dfa_accepting = {q for q in range(
                K) if accepting[q].solution_value() > 0.5}
            return True, dfa_transitions, dfa_accepting

        return False, None, None

    except Exception as e:
        print(f"Error during solving: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None


def verify_dfa(transitions, accepting_states, S, T):
    """Verify the DFA works correctly"""
    errors = []

    def process_string(string):
        state = 0
        for c in string:
            if (state, c) not in transitions:
                return None
            state = transitions[(state, c)]
        return state

    for s in S:
        final_state = process_string(s)
        if final_state is None or final_state not in accepting_states:
            return False, ["Acceptance verification failed"]

    for t in T:
        final_state = process_string(t)
        if final_state is None or final_state in accepting_states:
            return False, ["Rejection verification failed"]

    return True, []


def find_minimum_k(alphabet, S, T, max_k):
    """Find the minimum K for which a solution exists"""
    print(f"Searching for minimum K between 1 and {max_k}...")

    best_k = None
    best_transitions = None
    best_accepting = None

    estimated_min_k = max(1, min(int(math.log2(len(S) + len(T) + 1)), max_k))

    if len(S) + len(T) > 40:
        low, high = max(1, estimated_min_k), max_k
        while low <= high:
            mid = (low + high) // 2
            is_possible, transitions, accepting = solve_for_specific_k(
                alphabet, S, T, mid, time_limit=int(20000))

            if is_possible:
                best_k = mid
                best_transitions = transitions
                best_accepting = accepting
                high = mid - 1
            else:
                low = mid + 1

        if best_k is not None:
            is_valid, _ = verify_dfa(best_transitions, best_accepting, S, T)
            if is_valid:
                return best_k, best_transitions, best_accepting
            best_k = None

    start_k = estimated_min_k if best_k is None else max(1, best_k - 1)

    if len(S) + len(T) <= 40:
        for k in range(start_k, 0, -1):
            is_possible, transitions, accepting = solve_for_specific_k(
                alphabet, S, T, k)
            if is_possible:
                is_valid, _ = verify_dfa(transitions, accepting, S, T)
                if is_valid:
                    return k, transitions, accepting

    for k in range(start_k, max_k + 1):
        time_limit = int(min(20000 + k * 5000, 60000))
        is_possible, transitions, accepting = solve_for_specific_k(
            alphabet, S, T, k, time_limit=time_limit)

        if is_possible:
            is_valid, _ = verify_dfa(transitions, accepting, S, T)
            if is_valid:
                return k, transitions, accepting

    return best_k, best_transitions, best_accepting


def visualize_dfa(transitions, accepting_states, alphabet, K):
    """Simple DFA visualization"""
    alphabet_widths = [max(len(a), 3) for a in alphabet]

    header = "STATE | " + " | ".join(f"{a:^{w}}" for a, w in zip(
        alphabet, alphabet_widths)) + " | ACCEPTING"
    separator = "-" * len(header)

    rows = [header, separator]

    for q in range(K):
        row = [f" {q:4d} |"]
        for i, a in enumerate(alphabet):
            if (q, a) in transitions:
                row.append(f" {transitions[(q, a)]:^{alphabet_widths[i]}} |")
            else:
                row.append(f" {'-':^{alphabet_widths[i]}} |")

        row.append(" YES" if q in accepting_states else " NO")
        rows.append("".join(row))

    return "\n".join(rows)


def analyze_dfa_properties(transitions, accepting_states, K, alphabet):
    """Analyze the DFA properties"""
    properties = []
    properties.append(f"Initial state: 0")
    properties.append(f"Accepting states: {sorted(accepting_states)}")
    properties.append(
        f"Non-accepting states: {sorted(set(range(K)) - accepting_states)}")

    reachable = {0}
    frontier = {0}

    while frontier:
        new_frontier = set()
        for q in frontier:
            for a in alphabet:
                if (q, a) in transitions:
                    next_state = transitions[(q, a)]
                    if next_state not in reachable:
                        reachable.add(next_state)
                        new_frontier.add(next_state)
        frontier = new_frontier

    unreachable = set(range(K)) - reachable
    if unreachable:
        properties.append(f"Unreachable states: {sorted(unreachable)}")

    return properties


def trace_sample_strings(transitions, accepting_states, S, T):
    """Show how sample strings are processed"""
    traces = []
    samples = []

    if S:
        samples.append((min(S, key=len), True))
        samples.append((max(S, key=len), True))

    if T:
        samples.append((min(T, key=len), False))
        samples.append((max(T, key=len), False))

    if len(S) > 2:
        samples.append((S[len(S)//2], True))

    if len(T) > 2:
        samples.append((T[len(T)//2], False))

    for string, is_accept in samples:
        path = ["0"]
        current_state = 0

        for char in string:
            if (current_state, char) in transitions:
                current_state = transitions[(current_state, char)]
                path.append(str(current_state))
            else:
                path.append("?")
                break

        is_accepted = current_state in accepting_states
        status = "ACCEPT" if is_accepted else "REJECT"
        expected = "ACCEPT" if is_accept else "REJECT"

        trace_str = f"String '{string}': {' -> '.join(path)} ({status})"
        traces.append(trace_str)

    return traces


def main():
    """Main function to run the MIFSA solver"""
    start_time = time.time()

    try:
        input_file = "data.txt"

        if len(sys.argv) > 1:
            input_file = sys.argv[1]

        override_max_k = None
        if len(sys.argv) > 2:
            try:
                override_max_k = int(sys.argv[2])
            except ValueError:
                pass

        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found")
            return 1

        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

            alphabet = lines[0].split()
            K = int(lines[1])

            if override_max_k is not None:
                K = override_max_k

            S_count = int(lines[2])
            S = lines[3:3+S_count]

            current_line = 3 + S_count
            T_count = int(lines[current_line])
            T = lines[current_line+1:current_line+1+T_count]

        S_set = set(S)
        T_set = set(T)

        conflicts = S_set.intersection(T_set)
        if conflicts:
            print(f"Error: {len(conflicts)} strings in both S and T")
            return 1

        if len(S_set) < len(S) or len(T_set) < len(T):
            S = list(S_set)
            T = list(T_set)

        S.sort(key=len)
        T.sort(key=len)

        print(f"=== PROBLEM INSTANCE ===")
        print(f"Input file: {input_file}")
        print(f"Alphabet: {alphabet}")
        print(f"Maximum K: {K}")
        print(f"Accept strings: {len(S)}")
        print(f"Reject strings: {len(T)}")

        min_k, min_transitions, min_accepting_states = find_minimum_k(
            alphabet, S, T, max_k=K)

        if min_k is not None:
            print(f"\n=== SOLUTION FOUND ===")
            print(f"Minimum K = {min_k}")

            for q in range(min_k):
                for a in alphabet:
                    if (q, a) not in min_transitions:
                        min_transitions[(q, a)] = 0

            print(f"\n=== DFA TRANSITION TABLE ===")
            print(visualize_dfa(min_transitions,
                  min_accepting_states, alphabet, min_k))

            print(f"\n=== DFA PROPERTIES ===")
            for prop in analyze_dfa_properties(min_transitions, min_accepting_states, min_k, alphabet):
                print(f"* {prop}")

            output_file = f"{input_file.rsplit('.', 1)[0]}_solution_k{min_k}.txt"
            with open(output_file, 'w') as f:
                f.write(f"=== MINIMUM DFA SOLUTION (K={min_k}) ===\n\n")
                f.write(visualize_dfa(min_transitions,
                        min_accepting_states, alphabet, min_k))
                f.write("\n\n=== TRANSITION FUNCTION ===\n")
                for (q, a), q_prime in sorted(min_transitions.items()):
                    f.write(f"d({q}, {a}) = {q_prime}\n")
                f.write("\n=== ACCEPTING STATES ===\n")
                f.write(f"{sorted(min_accepting_states)}\n")

            print(f"\nSolution saved to {output_file}")
        else:
            print("\n=== NO SOLUTION FOUND ===")

        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        input("\nPress Enter to exit...")
        return 0

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
