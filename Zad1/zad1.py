#!/usr/bin/env python3
"""
Final AutID Solver - Uses a completely different approach to encode DFA constraints for Z3
"""
import z3
import sys
import time
import argparse


def read_data(filepath):
    """
    Read data from file in the specified format
    """
    with open(filepath, 'r') as f:
        # Read alphabet
        alphabet_line = f.readline().strip()
        if alphabet_line.startswith("Alphabet:"):
            alphabet = alphabet_line[len("Alphabet:"):].strip().split()
        else:
            alphabet = alphabet_line.strip().split()

        # Read k_max
        k_max = int(f.readline().strip())

        # Read accepting strings
        num_accepting = int(f.readline().strip())
        accepting = []
        for _ in range(num_accepting):
            accepting.append(f.readline().strip())

        # Read rejecting strings
        num_rejecting = int(f.readline().strip())
        rejecting = []
        for _ in range(num_rejecting):
            rejecting.append(f.readline().strip())

    return alphabet, k_max, accepting, rejecting


def solve_automaton(alphabet, accepting, rejecting, num_states, timeout=600):
    """
    Completely rewritten approach that avoids using Z3 expressions as dictionary keys.
    """
    print(
        f"Attempting to find DFA with {num_states} states (timeout: {timeout}s)...")

    # Set timeout for Z3
    z3.set_param('timeout', timeout * 1000)  # in milliseconds

    # Create solver
    solver = z3.Solver()

    # Create transition function as a 2D array (state x symbol)
    # delta[q][a_idx] represents the state reached from q on input alphabet[a_idx]
    delta = []
    for q in range(num_states):
        row = []
        for a_idx in range(len(alphabet)):
            var = z3.Int(f"delta_{q}_{a_idx}")
            solver.add(var >= 0)
            solver.add(var < num_states)
            row.append(var)
        delta.append(row)

    # Create final state variables
    is_final = [z3.Bool(f"final_{q}") for q in range(num_states)]

    # Map alphabet symbols to indices for easier reference
    alpha_to_idx = {a: i for i, a in enumerate(alphabet)}

    # Helper function to create constraints for a string
    def add_string_constraints(string, should_accept):
        # For empty string, constrain initial state
        if not string:
            if should_accept:
                solver.add(is_final[0])
            else:
                solver.add(z3.Not(is_final[0]))
            return

        # For non-empty strings, compute the end state
        end_state = compute_end_state(string)

        # Add constraint for final state
        if should_accept:
            solver.add(is_final[end_state])
        else:
            solver.add(z3.Not(is_final[end_state]))

    # Helper function to compute end state for a string
    def compute_end_state(string):
        # Start from initial state (0)
        current = z3.IntVal(0)

        # Process each symbol in the string
        for symbol in string:
            # Get index of the symbol in our alphabet
            symbol_idx = alpha_to_idx[symbol]

            # Create a new variable for the result of delta[current][symbol_idx]
            # We use the Select operation to access an element of delta based on current
            current = z3.Select(z3.Select(z3.K(z3.IntSort(), z3.IntSort(), z3.IntSort()),
                                          current),
                                z3.IntVal(symbol_idx))

            # Add constraints to define this select operation for concrete values
            for q in range(num_states):
                solver.add(z3.Select(z3.Select(z3.K(z3.IntSort(), z3.IntSort(), z3.IntSort()),
                                               z3.IntVal(q)),
                                     z3.IntVal(symbol_idx)) == delta[q][symbol_idx])

        return current

    # Add constraints for accepting strings
    for string in accepting:
        add_string_constraints(string, True)

    # Add constraints for rejecting strings
    for string in rejecting:
        add_string_constraints(string, False)

    # Check for solution
    start_time = time.time()
    result = solver.check()
    solve_time = time.time() - start_time

    print(
        f"Z3 solver finished in {solve_time:.2f} seconds with result: {result}")

    if result == z3.sat:
        model = solver.model()

        # Extract the transition function
        transitions = {}
        for q in range(num_states):
            transitions[q] = {}
            for a_idx, a in enumerate(alphabet):
                transitions[q][a] = model[delta[q][a_idx]].as_long()

        # Extract the final states
        final_states = set()
        for q in range(num_states):
            if model[is_final[q]] is not None and z3.is_true(model[is_final[q]]):
                final_states.add(q)

        return transitions, final_states

    return None


def solve_dfa_iterative(alphabet, accepting, rejecting, num_states, timeout=600):
    """
    A simpler approach that directly encodes the end state for each string.
    """
    print(
        f"Attempting to find DFA with {num_states} states (timeout: {timeout}s)...")

    # Set timeout for Z3
    z3.set_param('timeout', timeout * 1000)  # in milliseconds

    # Create solver
    solver = z3.Solver()

    # Create transition function
    delta = {}
    for q in range(num_states):
        delta[q] = {}
        for a_idx, a in enumerate(alphabet):
            delta[q][a] = z3.Int(f"delta_{q}_{a}")
            solver.add(delta[q][a] >= 0)
            solver.add(delta[q][a] < num_states)

    # Create final state variables
    is_final = [z3.Bool(f"final_{q}") for q in range(num_states)]

    # For each string, directly compute its end state using nested if-then-else expressions
    for string in accepting:
        if not string:
            # Empty string - initial state must be accepting
            solver.add(is_final[0])
            continue

        # For non-empty strings, compute end state directly
        end_state_expr = z3.IntVal(0)  # Start with initial state
        for symbol in string:
            # Build a nested if-then-else to compute next state
            ite_expr = z3.IntVal(0)  # Default
            for q in range(num_states):
                ite_expr = z3.If(end_state_expr == q,
                                 delta[q][symbol], ite_expr)
            end_state_expr = ite_expr

        # Build final constraint as a big disjunction
        constraint = z3.Or([z3.And(end_state_expr == q, is_final[q])
                           for q in range(num_states)])
        solver.add(constraint)

    # Same for rejecting strings
    for string in rejecting:
        if not string:
            # Empty string - initial state must be rejecting
            solver.add(z3.Not(is_final[0]))
            continue

        # For non-empty strings, compute end state directly
        end_state_expr = z3.IntVal(0)  # Start with initial state
        for symbol in string:
            # Build a nested if-then-else to compute next state
            ite_expr = z3.IntVal(0)  # Default
            for q in range(num_states):
                ite_expr = z3.If(end_state_expr == q,
                                 delta[q][symbol], ite_expr)
            end_state_expr = ite_expr

        # Build final constraint as a big disjunction
        constraint = z3.Or(
            [z3.And(end_state_expr == q, z3.Not(is_final[q])) for q in range(num_states)])
        solver.add(constraint)

    # Add symmetry breaking
    for i in range(1, num_states):
        used_i = z3.Or([delta[q][a] == i for q in range(i) for a in alphabet])

        # If state i is not used, then no state > i should be used
        if i < num_states - 1:
            not_used_higher = z3.And([
                z3.Not(z3.Or([delta[q][a] == j for q in range(i)
                       for a in alphabet]))
                for j in range(i+1, num_states)
            ])
            solver.add(z3.Implies(z3.Not(used_i), not_used_higher))

    # Check for solution
    start_time = time.time()
    result = solver.check()
    solve_time = time.time() - start_time

    print(
        f"Z3 solver finished in {solve_time:.2f} seconds with result: {result}")

    if result == z3.sat:
        model = solver.model()

        # Extract the transition function
        transitions = {}
        for q in range(num_states):
            transitions[q] = {}
            for a in alphabet:
                transitions[q][a] = model[delta[q][a]].as_long()

        # Extract the final states
        final_states = set()
        for q in range(num_states):
            if model[is_final[q]] is not None and z3.is_true(model[is_final[q]]):
                final_states.add(q)

        return transitions, final_states

    return None


def verify_dfa(alphabet, transitions, final_states, accepting, rejecting):
    """
    Verify that the constructed DFA correctly accepts/rejects all strings.
    """
    initial_state = 0
    errors = []

    # Check accepting strings
    for string in accepting:
        current_state = initial_state
        for symbol in string:
            if symbol not in alphabet:
                errors.append(f"Symbol '{symbol}' not in alphabet")
                continue
            current_state = transitions[current_state][symbol]

        if current_state not in final_states:
            errors.append(
                f"DFA incorrectly rejects accepting string: '{string}'")

    # Check rejecting strings
    for string in rejecting:
        current_state = initial_state
        for symbol in string:
            if symbol not in alphabet:
                errors.append(f"Symbol '{symbol}' not in alphabet")
                continue
            current_state = transitions[current_state][symbol]

        if current_state in final_states:
            errors.append(
                f"DFA incorrectly accepts rejecting string: '{string}'")

    return errors


def visualize_dfa(alphabet, transitions, final_states, k):
    """
    Create a textual visualization of the DFA.
    """
    result = []
    result.append("DFA Visualization:")
    result.append(f"Number of states: {k}")
    result.append(f"Initial state: 0")
    result.append(f"Final states: {', '.join(map(str, sorted(final_states)))}")
    result.append("\nTransition Table:")

    header = "State | " + " | ".join(alphabet) + " |"
    result.append(header)
    result.append("-" * len(header))

    for q in range(k):
        state_marker = "*" if q in final_states else " "
        row = f"{state_marker}{q:4d} | "
        row += " | ".join(str(transitions[q][a]) for a in alphabet)
        row += " |"
        result.append(row)

    return "\n".join(result)


def generate_dot_file(alphabet, transitions, final_states, k, filename="dfa.dot"):
    """
    Generate a DOT file for visualization with Graphviz
    """
    with open(filename, 'w') as f:
        f.write("digraph DFA {\n")
        f.write("    rankdir=LR;\n")
        f.write("    size=\"8,5\";\n")

        # Mark the initial state
        f.write("    node [shape = point]; qi;\n")

        # Define all states
        f.write("    node [shape = circle];\n")
        for q in range(k):
            if q in final_states:
                f.write(f"    {q} [shape = doublecircle];\n")

        # Initial state arrow
        f.write(f"    qi -> 0;\n")

        # Add transitions
        for q in range(k):
            for a in alphabet:
                next_state = transitions[q][a]
                f.write(f"    {q} -> {next_state} [label = \"{a}\"];\n")

        f.write("}\n")

    print(f"DOT file generated: {filename}")
    print("To visualize, install Graphviz and run: dot -Tpng dfa.dot -o dfa.png")


def run_dfa_on_string(alphabet, transitions, final_states, input_string):
    """
    Run the constructed DFA on a given input string and return if it's accepted.
    """
    current_state = 0  # Initial state
    path = [current_state]

    for symbol in input_string:
        if symbol not in alphabet:
            return False, f"Symbol '{symbol}' not in alphabet", path
        current_state = transitions[current_state][symbol]
        path.append(current_state)

    return current_state in final_states, "Accepted" if current_state in final_states else "Rejected", path


def main():
    parser = argparse.ArgumentParser(
        description='Automaton Identification Solver')
    parser.add_argument('input_file', nargs='?',
                        help='Path to the input data file')
    parser.add_argument('--min-states', type=int, default=1,
                        help='Minimum number of states to try')
    parser.add_argument('--max-states', type=int,
                        help='Maximum number of states to try (overrides file value)')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout in seconds for each state count')
    parser.add_argument('--exact', type=int,
                        help='Try only this exact number of states')
    args = parser.parse_args()

    # Get input file path
    if args.input_file:
        filepath = args.input_file
    else:
        filepath = input("Enter the path to the data file: ")

    try:
        # Read data
        alphabet, k_max, accepting, rejecting = read_data(filepath)

        print(f"Alphabet: {alphabet}")
        print(f"Maximum states from file: {k_max}")
        if args.max_states:
            print(f"Maximum states from command line: {args.max_states}")
        print(f"Accepting strings: {accepting}")
        print(f"Rejecting strings: {rejecting}")

        print(
            f"\nProcessing {len(accepting)} accepting strings and {len(rejecting)} rejecting strings...")

        # If exact number of states is specified, only try that
        if args.exact:
            result = solve_dfa_iterative(
                alphabet, accepting, rejecting, args.exact, args.timeout)

            if result:
                transitions, final_states = result
                k = args.exact
                print(f"\nSuccess! Found a DFA with {k} states.")

                # Verify and display
                errors = verify_dfa(alphabet, transitions,
                                    final_states, accepting, rejecting)
                if errors:
                    print("\nWARNING: DFA verification failed!")
                    for error in errors:
                        print(f"  - {error}")
                else:
                    print(
                        "Verification successful: DFA correctly handles all input strings.")

                print("\n" + visualize_dfa(alphabet, transitions, final_states, k))
                generate_dot_file(alphabet, transitions, final_states, k)

                # Interactive mode
                print(
                    "\nInteractive mode: Test strings against the DFA (type 'exit' to quit)")
                while True:
                    test_string = input("\nEnter a string to test: ")
                    if test_string.lower() == 'exit':
                        break

                    accepted, result_msg, path = run_dfa_on_string(
                        alphabet, transitions, final_states, test_string)
                    print(f"Result: {result_msg}")
                    print(f"Path: {' -> '.join(map(str, path))}")
            else:
                print(f"No solution found with exactly {args.exact} states.")
            return

        # Otherwise, try increasing number of states
        min_states = args.min_states
        max_states = args.max_states if args.max_states else k_max

        for k in range(min_states, max_states + 1):
            try:
                # Try the iterative approach which should be more reliable
                result = solve_dfa_iterative(
                    alphabet, accepting, rejecting, k, args.timeout)

                if result:
                    transitions, final_states = result
                    print(f"\nSuccess! Found a DFA with {k} states.")

                    # Verify the DFA
                    errors = verify_dfa(
                        alphabet, transitions, final_states, accepting, rejecting)
                    if errors:
                        print("\nWARNING: DFA verification failed!")
                        print(f"Found {len(errors)} errors. First 10:")
                        for i, error in enumerate(errors[:10]):
                            print(f"  - {error}")
                        if len(errors) > 10:
                            print(f"  ... and {len(errors) - 10} more errors")
                    else:
                        print(
                            "Verification successful: DFA correctly handles all input strings.")

                    # Display the DFA
                    print("\n" + visualize_dfa(alphabet,
                          transitions, final_states, k))

                    # Output formal specification
                    print("\nFormal DFA Specification:")
                    print(f"Q = {{{', '.join(str(i) for i in range(k))}}}")
                    print(f"Σ = {{{', '.join(alphabet)}}}")
                    print(f"q₀ = 0")
                    print(f"F = {{{', '.join(str(i) for i in final_states)}}}")
                    print("δ (transition function):")
                    for q in range(k):
                        for a in alphabet:
                            print(f"  δ({q}, {a}) = {transitions[q][a]}")

                    # Generate DOT file
                    generate_dot_file(alphabet, transitions, final_states, k)

                    # Interactive mode
                    print(
                        "\nInteractive mode: Test strings against the DFA (type 'exit' to quit)")
                    while True:
                        test_string = input("\nEnter a string to test: ")
                        if test_string.lower() == 'exit':
                            break

                        accepted, result_msg, path = run_dfa_on_string(
                            alphabet, transitions, final_states, test_string)
                        print(f"Result: {result_msg}")
                        print(f"Path: {' -> '.join(map(str, path))}")

                    return
            except Exception as e:
                print(f"Error with {k} states: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nNo solution found with up to {max_states} states.")
        print("Suggestions:")
        print("  1. Try increasing the maximum number of states")
        print("  2. Check for contradictions in your example strings")
        print("  3. Increase the timeout for the solver")
        print("  4. Try just searching for the 8-state solution with --exact 8")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
