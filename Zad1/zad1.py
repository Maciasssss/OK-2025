from ortools.linear_solver import pywraplp
import sys
import time
import re
from collections import defaultdict

######################
# TRIE IMPLEMENTATION
######################


class TrieNode:
    def __init__(self):
        self.children = {}  # key: symbol, value: TrieNode
        # List of booleans for strings ending at this node (True = accept, False = reject)
        self.outcomes = []


def build_trie(accepted, rejected):
    root = TrieNode()
    # Track which strings have been inserted
    inserted = set()

    def insert(word, outcome):
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        # Check for conflicts
        if word in inserted and (outcome not in node.outcomes):
            print(f"Warning: Conflicting outcomes for string '{word}'")
        node.outcomes.append(outcome)
        inserted.add(word)

    for s in accepted:
        insert(s, True)
    for t in rejected:
        insert(t, False)
    return root

##############################
# PARSING AND PREPARATION CODE
##############################


def parse_input_file(input_file):
    """Parse input file with robust error handling."""
    try:
        with open(input_file, 'r') as f:
            content = f.read().strip()

        # Split lines and remove empty lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Parse alphabet (Note: header "Alfabet:" is used)
        alphabet_match = re.match(
            r'^Alfabet:\s*([^\s]+)', lines[0], re.IGNORECASE)
        if alphabet_match:
            alphabet = list(alphabet_match.group(1))
        else:
            alphabet = ['0', '1']  # default alphabet

        # Parse K (max states)
        k_match = re.match(r'^maximum_k_value:\s*(\d+)',
                           lines[1], re.IGNORECASE)
        K = int(k_match.group(1)) if k_match else 10

        # Parse accepted and rejected strings
        accepted = []
        rejected = []
        current_section = None
        for line in lines[2:]:
            if re.match(r'^accepting_strings_S:', line, re.IGNORECASE):
                current_section = 'S'
                continue
            elif re.match(r'^rejecting_strings_T:', line, re.IGNORECASE):
                current_section = 'T'
                continue
            if current_section is None:
                continue
            # Split comma-separated strings
            strings = [s.strip() for s in line.split(',') if s.strip()]
            valid_strings = [s for s in strings if all(
                ch in alphabet for ch in s)]
            if current_section == 'S':
                accepted.extend(valid_strings)
            else:
                rejected.extend(valid_strings)

        print(f"\n=== Parsed Problem Instance ===")
        print(f"Alphabet: {alphabet}")
        print(f"Maximum K: {K}")
        print(f"Accept strings: {len(accepted)}")
        print(f"Reject strings: {len(rejected)}")
        return alphabet, K, accepted, rejected

    except Exception as e:
        print(f"Error parsing input file: {e}")
        return None, None, None, None

#################################
# MILP SOLVER WITH TRIE GROUPING
#################################


class DFAConstructionSolver:
    def __init__(self, alphabet, max_states, accept_strings, reject_strings):
        self.alphabet = alphabet
        self.max_states = max_states
        self.accept_strings = accept_strings
        self.reject_strings = reject_strings
        # Build a trie that groups common prefixes.
        self.trie_root = build_trie(accept_strings, reject_strings)

    def solve_with_milp(self):
        solver = pywraplp.Solver.CreateSolver(
            'SCIP_MIXED_INTEGER_PROGRAMMING')
        if not solver:
            print("MILP solver not available.")
            return None, None

        # Set solver parameters (adjust as needed)
        solver.SetNumThreads(8)
        solver.SetTimeLimit(300000)  # 5 minutes
        # (Additional parameters might be set if using a commercial solver like Gurobi/CPLEX)

        K = self.max_states
        alphabet = self.alphabet

        # 1. Transition variables:
        # For each state q and symbol a, T_vars[(q,a,q2)]=1 if the DFA transitions from state q to state q2 on symbol a.
        T_vars = {}
        for q in range(K):
            for a in alphabet:
                vars_for_transition = []
                for q2 in range(K):
                    T_vars[(q, a, q2)] = solver.IntVar(0, 1, f"T_{q}_{a}_{q2}")
                    vars_for_transition.append(T_vars[(q, a, q2)])
                solver.Add(solver.Sum(vars_for_transition) == 1)

        # 2. Accepting state variables:
        is_accepting = {}
        for q in range(K):
            is_accepting[q] = solver.IntVar(0, 1, f"is_accepting_{q}")
        if self.accept_strings:
            solver.Add(solver.Sum([is_accepting[q] for q in range(K)]) >= 1)

        # 3. Simulation constraints using the trie.
        # For each trie node, we will have a variable that indicates the state of the DFA after reading that prefix.
        # We use a dictionary mapping node IDs to a dictionary of state indicator variables.
        x_vars = {}  # key: node id, value: dict mapping state q to variable

        # To traverse the trie, assign each node a unique id.
        node_id_counter = [0]

        def assign_ids(node):
            node.id = node_id_counter[0]
            node_id_counter[0] += 1
            for child in node.children.values():
                assign_ids(child)
        assign_ids(self.trie_root)

        # Create x_vars for each node in the trie.
        def create_x_vars(node):
            x_vars[node.id] = {}
            for q in range(K):
                x_vars[node.id][q] = solver.IntVar(
                    0, 1, f"x_node{node.id}_{q}")
            for child in node.children.values():
                create_x_vars(child)
        create_x_vars(self.trie_root)

        # The root must be state 0.
        solver.Add(x_vars[self.trie_root.id][0] == 1)
        for q in range(1, K):
            solver.Add(x_vars[self.trie_root.id][q] == 0)
        # At each node, exactly one state is active.
        for node_id, state_vars in x_vars.items():
            solver.Add(solver.Sum([state_vars[q] for q in range(K)]) == 1)

        # Now, add transition simulation constraints along each trie edge.
        # For each node and each child corresponding to symbol 'a', simulate transition.
        y_vars = {}  # auxiliary variables to linearize product.

        def add_transition_constraints(parent):
            parent_id = parent.id
            for a, child in parent.children.items():
                child_id = child.id
                # For each possible transition from parent's state q to child's state q2 on symbol a.
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
                # Enforce that the child node's state variable is the sum over y_vars.
                for q2 in range(K):
                    solver.Add(x_vars[child_id][q2] ==
                               solver.Sum([y_vars[(parent_id, child_id, q, q2)] for q in range(K)]))
                # Recursively add constraints for the child.
                add_transition_constraints(child)
        add_transition_constraints(self.trie_root)

        # 4. Final-state constraints: For every trie node that is a leaf (has outcomes), enforce the outcome.
        def add_final_constraints(node):
            if node.outcomes:
                for q in range(K):
                    # If this node contains any accepted strings and the DFA is in state q,
                    # then state q must be accepting
                    if any(outcome for outcome in node.outcomes):
                        # x_vars[node.id][q] = 1 => is_accepting[q] = 1
                        # This is equivalent to x_vars[node.id][q] <= is_accepting[q]
                        solver.Add(x_vars[node.id][q] <= is_accepting[q])

                    # If this node contains any rejected strings and the DFA is in state q,
                    # then state q must be non-accepting
                    if any(not outcome for outcome in node.outcomes):
                        # x_vars[node.id][q] = 1 => is_accepting[q] = 0
                        # This is equivalent to x_vars[node.id][q] <= (1 - is_accepting[q])
                        solver.Add(x_vars[node.id][q] <= 1 - is_accepting[q])

            # Recursively process children
            for child in node.children.values():
                add_final_constraints(child)

        # 5. Dummy objective (feasibility only)
        solver.Minimize(0)

        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Extract the transition function.
            extracted_transitions = {}
            for q in range(K):
                for a in alphabet:
                    for q2 in range(K):
                        if T_vars[(q, a, q2)].solution_value() > 0.5:
                            extracted_transitions[(q, a)] = q2
                            break
            extracted_accepting = {q for q in range(
                K) if is_accepting[q].solution_value() > 0.5}
            return extracted_transitions, extracted_accepting
        else:
            print("No solution found.")
            return None, None


def main():
    start_time = time.time()
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data.txt"
    alphabet, K, accepted, rejected = parse_input_file(input_file)
    if alphabet is None:
        return 1

    solver_instance = DFAConstructionSolver(alphabet, K, accepted, rejected)
    transitions, accepting = solver_instance.solve_with_milp()

    if transitions is not None and accepting is not None:
        print("\n=== SOLUTION FOUND ===")
        print("Transition Function:")
        for (state, symbol), next_state in sorted(transitions.items()):
            print(f"δ({state}, {symbol}) = {next_state}")
        print("\nAccepting States:", sorted(accepting))
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        return 0
    else:
        print("\n=== NO SOLUTION FOUND ===")
        return 1


if __name__ == "__main__":
    main()
