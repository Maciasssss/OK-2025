from ortools.linear_solver import pywraplp
import sys
import time
import re
from collections import defaultdict


class TrieNode:
    def __init__(self):
        self.children = {}
        self.outcomes = []


def build_trie(accepted, rejected):
    root = TrieNode()

    def insert(word, outcome):
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        if outcome not in node.outcomes:
            node.outcomes.append(outcome)

    for s in accepted:
        insert(s, True)
    for t in rejected:
        insert(t, False)
    return root


def parse_input_file(input_file):
    """Parse input file with improved error handling for incorrect format."""
    try:
        with open(input_file, 'r') as f:
            content = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None, None, None, None
    except PermissionError:
        print(f"Error: Permission denied for input file '{input_file}'.")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return None, None, None, None

    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        print("Error: Input file is empty.")
        return None, None, None, None

    # Check file structure
    if len(lines) < 3:
        print("Error: Input file is missing required sections (Alphabet, K value, and strings).")
        return None, None, None, None

    # Parse alphabet - accept both "Alphabet" and "Alfabet" with comma-separated symbols
    alphabet_pattern = r'^(?:Alphabet|Alfabet):\s*(.*?)$'
    alphabet_match = re.match(alphabet_pattern, lines[0], re.IGNORECASE)

    if not alphabet_match:
        print(
            "Error: Cannot parse alphabet. Expected 'Alphabet: [symbols]' or 'Alfabet: [symbols]'.")
        return None, None, None, None

    # Extract and process comma-separated symbols
    raw_alphabet = alphabet_match.group(1)

    # Split by commas and clean up each symbol
    alphabet = [symbol.strip()
                for symbol in raw_alphabet.split(',') if symbol.strip()]

    if not alphabet:
        print("Error: No valid alphabet symbols found.")
        return None, None, None, None

    # Check for duplicate symbols in alphabet
    if len(alphabet) != len(set(alphabet)):
        print("Warning: Alphabet contains duplicate symbols. Using unique symbols only.")
        alphabet = list(set(alphabet))

    # Parse K value
    k_match = re.match(r'^maximum_k_value:\s*(\d+)', lines[1], re.IGNORECASE)
    if not k_match:
        print(
            "Error: Cannot parse maximum K value. Expected 'maximum_k_value: [number]'.")
        return None, None, None, None

    try:
        K = int(k_match.group(1))
        if K <= 0:
            print("Error: Maximum K value must be positive.")
            return None, None, None, None
    except ValueError:
        print("Error: Maximum K value must be an integer.")
        return None, None, None, None

    # Parse accepting and rejecting strings
    accepted = []
    rejected = []
    current_section = None
    found_accept_section = False
    found_reject_section = False

    for line in lines[2:]:
        if re.match(r'^accepting_strings_S:', line, re.IGNORECASE):
            current_section = 'S'
            found_accept_section = True
            continue
        elif re.match(r'^rejecting_strings_T:', line, re.IGNORECASE):
            current_section = 'T'
            found_reject_section = True
            continue

        if current_section is None:
            continue

        # Split comma-separated strings
        strings = [s.strip() for s in line.split(',') if s.strip()]

        # Validate strings against alphabet
        invalid_strings = []
        valid_strings = []
        for s in strings:
            # Check if all characters in the string are in the alphabet
            if all(ch in alphabet for ch in s):
                valid_strings.append(s)
            else:
                invalid_strings.append(s)

        if invalid_strings:
            print(
                f"Warning: Ignoring {len(invalid_strings)} string(s) with characters not in alphabet.")
            for inv_str in invalid_strings[:5]:  # Show first 5 invalid strings
                print(
                    f"  - '{inv_str}' contains character(s) not in alphabet {alphabet}")
            if len(invalid_strings) > 5:
                print(f"  - ... and {len(invalid_strings) - 5} more")

        if current_section == 'S':
            accepted.extend(valid_strings)
        else:
            rejected.extend(valid_strings)

    # Check if we found the required sections
    if not found_accept_section:
        print("Warning: No 'accepting_strings_S' section found.")
    if not found_reject_section:
        print("Warning: No 'rejecting_strings_T' section found.")

    # Check for empty string handling
    if "" in accepted:
        print("Note: Empty string is in accepting set.")
    if "" in rejected:
        print("Note: Empty string is in rejecting set.")
    if "" in accepted and "" in rejected:
        print("Error: Empty string cannot be both accepted and rejected.")
        return None, None, None, None

    # Determine theoretical minimum k
    min_theoretical_k = 1
    if accepted and rejected:
        min_theoretical_k = 2

    print(f"\n=== Parsed Problem Instance ===")
    print(f"Alphabet: {alphabet}")
    print(f"Maximum K: {K} (will try from k={min_theoretical_k} to {K})")
    print(f"Accept strings: {len(accepted)}")
    print(f"Reject strings: {len(rejected)}")

    # Final validation - check if we have any strings to process
    if not accepted and not rejected:
        print("Error: No valid strings found in input file.")
        return None, None, None, None

    return alphabet, K, accepted, rejected


class DFAConstructionSolver:
    def __init__(self, alphabet, max_states, accept_strings, reject_strings):
        self.alphabet = alphabet
        self.max_states = max_states
        self.accept_strings = accept_strings
        self.reject_strings = reject_strings
        self.trie_root = build_trie(accept_strings, reject_strings)

    def solve_with_milp(self):
        # Early termination for impossible cases
        if self.max_states == 1 and self.accept_strings and self.reject_strings:
            print("Infeasible: K=1 cannot handle both accepting and rejecting strings")
            return None, None

        # Create solver and verify availability
        solver = pywraplp.Solver.CreateSolver('SCIP_MIXED_INTEGER_PROGRAMMING')
        if not solver:
            print(
                "Error: MILP solver not available. Please check your OR-Tools installation.")
            return None, None

        # Set solver parameters for better performance
        solver.SetNumThreads(8)
        solver.SetTimeLimit(300000)  # 5 minutes

        K = self.max_states
        alphabet = self.alphabet

        # Create transition variables
        T_vars = {}
        for q in range(K):
            for a in alphabet:
                vars_for_transition = []
                for q2 in range(K):
                    T_vars[(q, a, q2)] = solver.IntVar(0, 1, f"T_{q}_{a}_{q2}")
                    vars_for_transition.append(T_vars[(q, a, q2)])
                solver.Add(solver.Sum(vars_for_transition) == 1)

        # Create accepting state variables
        is_accepting = {}
        for q in range(K):
            is_accepting[q] = solver.IntVar(0, 1, f"is_accepting_{q}")
        if self.accept_strings:
            solver.Add(solver.Sum([is_accepting[q] for q in range(K)]) >= 1)

        # Create trie node state variables
        x_vars = {}
        node_id_counter = [0]

        def assign_ids(node):
            node.id = node_id_counter[0]
            node_id_counter[0] += 1
            for child in node.children.values():
                assign_ids(child)
        assign_ids(self.trie_root)

        def create_x_vars(node):
            x_vars[node.id] = {}
            for q in range(K):
                x_vars[node.id][q] = solver.IntVar(
                    0, 1, f"x_node{node.id}_{q}")
            for child in node.children.values():
                create_x_vars(child)
        create_x_vars(self.trie_root)

        # The root must be in state 0
        solver.Add(x_vars[self.trie_root.id][0] == 1)
        for q in range(1, K):
            solver.Add(x_vars[self.trie_root.id][q] == 0)

        # At each node, exactly one state is active
        for node_id, state_vars in x_vars.items():
            solver.Add(solver.Sum([state_vars[q] for q in range(K)]) == 1)

        # Create transition simulation variables and constraints
        y_vars = {}

        def add_transition_constraints(parent):
            parent_id = parent.id
            for a, child in parent.children.items():
                child_id = child.id
                for q in range(K):
                    for q2 in range(K):
                        y_vars[(parent_id, child_id, q, q2)] = solver.IntVar(
                            0, 1, f"y_{parent_id}_{child_id}_{q}_{q2}")
                        # Linearize y = x(parent,q) * T_vars[(q,a,q2)]
                        solver.Add(
                            y_vars[(parent_id, child_id, q, q2)] <= x_vars[parent_id][q])
                        solver.Add(
                            y_vars[(parent_id, child_id, q, q2)] <= T_vars[(q, a, q2)])
                        solver.Add(y_vars[(parent_id, child_id, q, q2)] >=
                                   x_vars[parent_id][q] + T_vars[(q, a, q2)] - 1)

                # Enforce that the child node's state variable is the sum over y_vars
                for q2 in range(K):
                    solver.Add(x_vars[child_id][q2] ==
                               solver.Sum([y_vars[(parent_id, child_id, q, q2)] for q in range(K)]))

                add_transition_constraints(child)

        def add_final_constraints(node):
            if node.outcomes:
                # Check for conflicts in outcomes
                has_accepting = any(outcome for outcome in node.outcomes)
                has_rejecting = any(not outcome for outcome in node.outcomes)

                if has_accepting and has_rejecting:
                    print(f"Warning: Node {node.id} has conflicting outcomes")
                    solver.Add(solver.Sum([1]) > K)  # Force infeasibility
                    return

                # Add constraints based on outcomes
                for q in range(K):
                    if has_accepting:
                        solver.Add(x_vars[node.id][q] <= is_accepting[q])
                    if has_rejecting:
                        solver.Add(x_vars[node.id][q] <= 1 - is_accepting[q])

            # Process all children
            for child in node.children.values():
                add_final_constraints(child)

        # Add all constraints to the solver
        add_transition_constraints(self.trie_root)
        add_final_constraints(self.trie_root)

        # Set objective function (feasibility only)
        solver.Minimize(0)

        # Solve the model
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Extract the transition function
            extracted_transitions = {}
            for q in range(K):
                for a in alphabet:
                    for q2 in range(K):
                        if T_vars[(q, a, q2)].solution_value() > 0.5:
                            extracted_transitions[(q, a)] = q2
                            break

            # Extract accepting states
            extracted_accepting = {q for q in range(
                K) if is_accepting[q].solution_value() > 0.5}
            return extracted_transitions, extracted_accepting
        else:
            print("No solution found.")
            return None, None


def main():
    start_time = time.time()

    # Get input file path and optional k value from command line
    if len(sys.argv) < 2:
        print("No input file specified. Using default 'data.txt'.")
        input_file = "data.txt"
        specific_k = None
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        specific_k = None
    else:
        input_file = sys.argv[1]
        try:
            specific_k = int(sys.argv[2])
            if specific_k <= 0:
                print(
                    f"Warning: Invalid k value {specific_k}. K must be positive.")
                print("Falling back to incremental mode.")
                specific_k = None
        except ValueError:
            print(
                f"Warning: Could not parse '{sys.argv[2]}' as an integer k value.")
            print("Falling back to incremental mode.")
            specific_k = None

    # Parse input file
    alphabet, max_K, accepted, rejected = parse_input_file(input_file)
    if alphabet is None:
        print("Exiting due to input file errors.")
        return 1

    # Handle specific k value if provided
    if specific_k is not None:
        if specific_k > max_K:
            print(
                f"Warning: Specified k ({specific_k}) exceeds maximum in file ({max_K}).")
            print(f"Using k = {specific_k} anyway.")

        print(f"\n=== Running Solver for k = {specific_k} ===")
        k_start_time = time.time()

        solver_instance = DFAConstructionSolver(
            alphabet, specific_k, accepted, rejected)
        transitions, accepting = solver_instance.solve_with_milp()

        k_time = time.time() - k_start_time
        print(f"Time for k={specific_k}: {k_time:.2f} seconds")

        if transitions is not None and accepting is not None:
            print(f"✓ Solution found with {specific_k} states!")
            best_solution = (transitions, accepting)
            best_k = specific_k
        else:
            print(f"× No solution found with {specific_k} states.")
            best_solution = None
            best_k = None
    else:
        print(f"\n=== Starting Incremental Solver (k=2 to {max_K}) ===")

        best_solution = None
        best_k = None

        # Try each value of k from 2 to max_K
        for k in range(2, max_K + 1):
            print(f"\n--- Trying with k = {k} states ---")
            k_start_time = time.time()

            solver_instance = DFAConstructionSolver(
                alphabet, k, accepted, rejected)
            transitions, accepting = solver_instance.solve_with_milp()

            k_time = time.time() - k_start_time
            print(f"Time for k={k}: {k_time:.2f} seconds")

            if transitions is not None and accepting is not None:
                print(f"✓ Solution found with {k} states!")
                best_solution = (transitions, accepting)
                best_k = k
                break

    total_time = time.time() - start_time

    if best_solution is not None:
        transitions, accepting = best_solution
        print(f"\n=== SOLUTION FOUND WITH {best_k} STATES ===")
        print("Transition Function:")
        for (state, symbol), next_state in sorted(transitions.items()):
            print(f"δ({state}, {symbol}) = {next_state}")
        print("\nAccepting States:", sorted(accepting))
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        return 0
    else:
        if specific_k is not None:
            print(f"\n=== NO SOLUTION FOUND WITH {specific_k} STATES ===")
        else:
            print(f"\n=== NO SOLUTION FOUND FOR ANY K VALUE ===")
            print(f"Tried all k values from 2 to {max_K}")
        print(f"Total execution time: {total_time:.2f} seconds")
        return 1


if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    exit(exit_code)
