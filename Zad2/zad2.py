import re
import os
import math
import time
import json
import hashlib
import argparse
import clingo
from datetime import datetime
try:
    import sympy
    from sympy.ntheory import factorint
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    print("WARNING: SymPy not found. Install with 'pip install sympy' for better factorization.")

try:
    import gmpy2
    HAS_GMPY2 = True
except ImportError:
    HAS_GMPY2 = False

try:
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    HAS_MP = True
except ImportError:
    HAS_MP = False
    print("INFO: Multiprocessing not available. Parallel factorization disabled.")

CACHE_FILE = "factorization_cache.json"
MAX_WORKERS = max(1, mp.cpu_count() - 1) if HAS_MP else 1

GLOBAL_USE_CACHE = True

# ==============================================================================
# Factorization System
# ==============================================================================


def initialize_cache():
    """Initialize the cache file with an empty JSON object if it doesn't exist"""
    if not os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'w') as f:
                f.write('{}')
            return True
        except Exception:
            return False
    return True


def load_cache():
    """Load the factorization cache from disk with error handling"""
    if not os.path.exists(CACHE_FILE):
        initialize_cache()
        return {}

    try:
        with open(CACHE_FILE, 'r') as f:
            data = f.read().strip()
            if not data:
                return {}
            return json.loads(data)
    except Exception:
        initialize_cache()
        return {}


def save_cache(cache):
    """Save the factorization cache to disk"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        return True
    except Exception:
        return False


def number_hash(n):
    """Create a hash for a number to use as cache key"""
    return hashlib.md5(str(n).encode()).hexdigest()


def is_prime_fast(n):
    """Quick primality test using available libraries or fallback"""
    if HAS_SYMPY:
        return sympy.isprime(n)
    elif HAS_GMPY2:
        return gmpy2.is_prime(n, 25)
    else:
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True


def factorize_small(n):
    """Optimized factorization for smaller numbers"""
    factors = {}

    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    for i in range(3, 100, 2):
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i

    if n > 1:
        sqrt_n = int(math.sqrt(n))
        if sqrt_n * sqrt_n == n and is_prime_fast(sqrt_n):
            factors[sqrt_n] = 2
        else:
            for i in range(101, 10000, 2):
                if i * i > n:
                    break
                while n % i == 0:
                    factors[i] = factors.get(i, 0) + 1
                    n //= i

            if n > 1:
                factors[n] = factors.get(n, 0) + 1

    return factors


def factorize_with_sympy(n):
    """Use SymPy's advanced factorization"""
    return factorint(n, limit=10**7, use_trial=True, use_rho=True, use_pm1=True, verbose=False)


def factorize_with_gmpy2(n):
    """Use GMPY2's factorization if available"""
    if not HAS_GMPY2:
        return None

    factors = {}
    temp_n = gmpy2.mpz(n)

    i = gmpy2.mpz(2)
    while i * i <= temp_n:
        while temp_n % i == 0:
            factors[int(i)] = factors.get(int(i), 0) + 1
            temp_n //= i
        i += 1

    if temp_n > 1:
        factors[int(temp_n)] = factors.get(int(temp_n), 0) + 1

    return factors


def factorize_with_params(n):
    """Wrapper for multiprocessing - must be at module level"""
    return factorize_large_number(n, GLOBAL_USE_CACHE)


def factorize_large_number(n, use_cache=True):
    """
    Optimized factorization for large numbers with caching,
    algorithm selection, and fallbacks
    """
    if isinstance(n, str):
        n = int(n)

    cache = load_cache() if use_cache else {}
    n_hash = number_hash(n)
    if n_hash in cache:
        print(f"Cache hit for {n}")
        return {int(k): v for k, v in cache[n_hash].items()}

    print(f"Factorizing {n}...")
    start_time = time.time()

    if n < 10**9:
        factors = factorize_small(n)
    elif is_prime_fast(n):
        factors = {n: 1}
    else:
        if HAS_SYMPY:
            factors = factorize_with_sympy(n)
        elif HAS_GMPY2:
            factors = factorize_with_gmpy2(n)
        else:
            factors = factorize_small(n)

    end_time = time.time()
    print(f"Factorization completed in {end_time - start_time:.2f} seconds")

    if use_cache:
        cache[n_hash] = {str(k): v for k, v in factors.items()}
        save_cache(cache)

    return factors


def factorize_number_list(numbers, use_parallel=True, use_cache=True):
    """Factorize a list of numbers, potentially in parallel"""
    global GLOBAL_USE_CACHE
    GLOBAL_USE_CACHE = use_cache

    if not use_parallel or not HAS_MP or len(numbers) <= 1 or MAX_WORKERS <= 1:
        return [(i, factorize_large_number(n, use_cache)) for i, n in numbers]

    print(f"Using {MAX_WORKERS} workers for parallel factorization")
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(factorize_with_params, [n for _, n in numbers])

    return [(numbers[i][0], result) for i, result in enumerate(results)]


def parse_scientific_notation(num_str):
    """Handle scientific notation in input numbers"""
    num_str = num_str.strip()
    if 'e' in num_str.lower() or 'E' in num_str:
        return int(float(num_str))
    return int(num_str)


def format_factorization(factors):
    """Format a factorization dictionary in a readable way"""
    if not factors:
        return "No factorization available"

    terms = []
    for prime, exp in sorted(factors.items()):
        if exp == 1:
            terms.append(f"{prime}")
        else:
            terms.append(f"{prime}^{exp}")

    return " × ".join(terms)

# ==============================================================================
# ASP Program Generation and Solving
# ==============================================================================


def create_asp_program(input_numbers, number_factors, target_factors=None):
    """
    Create an ASP program for the subset product problem

    Args:
        input_numbers: Dictionary mapping indexes to values
        number_factors: Dictionary mapping indexes to their prime factorizations
        target_factors: Dictionary mapping prime factors to exponents for the target
    """
    # Basic program structure
    program = """
    % Subset Product Problem Solver
    
    % Each number is either selected or not
    { select(N) : input_number(N,_) }.
    
    % For each prime in the target, the sum of its powers must match exactly
    :- target_factor(P,TC), TC != #sum{ E,N : select(N), has_factor(N,P,E) }.
    
    % For primes not in the target, ensure they don't appear in the selection
    :- has_factor(_, P,_), not target_factor(P,_), #sum{ E,N : select(N), has_factor(N,P,E) } > 0.
    
    % Optimization: Minimize the number of selected elements
    #minimize { 1,N : select(N) }.
    
    % Output
    #show select/1.
    #show selected_value/1.
    """

    # Add input facts
    facts = []
    for idx, value in input_numbers.items():
        facts.append(f"input_number({idx}, {value}).")

    # Add factorization facts
    for idx, factors in number_factors.items():
        for prime, exponent in factors.items():
            facts.append(f"has_factor({idx}, {prime}, {exponent}).")

    # Add target factorization
    if target_factors:
        for prime, exponent in target_factors.items():
            facts.append(f"target_factor({prime}, {exponent}).")

    # Add selected_value rule
    facts.append("selected_value(V) :- select(N), input_number(N,V).")

    # Combine everything
    return program + "\n" + "\n".join(facts)


def solve_with_clingo(program, timeout=None):
    """Solve the ASP program using Clingo with timeout option"""
    control = clingo.Control()
    if timeout:
        control.configuration.solve.timeout = timeout

    control.add("base", [], program)
    control.ground([("base", [])])

    solution = None
    models = []

    def on_model(model):
        nonlocal solution
        solution = model.symbols(shown=True)
        models.append(model.symbols(shown=True))

    control.configuration.solve.models = 0
    handle = control.solve(on_model=on_model)

    if handle.satisfiable:
        # Return the last (optimal) model
        return models[-1] if models else None
    else:
        return None


def parse_solution(solution):
    """Parse the solution from Clingo into a readable format"""
    if not solution:
        return {"selected_indices": [], "selected_values": []}

    selected_indices = []
    selected_values = []

    for atom in solution:
        if atom.name == "select" and len(atom.arguments) == 1:
            selected_indices.append(atom.arguments[0].number)
        elif atom.name == "selected_value" and len(atom.arguments) == 1:
            selected_values.append(atom.arguments[0].number)

    return {
        "selected_indices": sorted(selected_indices),
        "selected_values": selected_values
    }

# ==============================================================================
# Input Parsing
# ==============================================================================


def parse_input_file(filename="data.txt"):
    """Parse the input file containing the set and target"""
    try:
        with open(filename, "r") as f:
            content = f.read()

        set_match = re.search(r'Set\s*=\s*{([^}]+)}', content)
        numbers = []
        if set_match:
            set_str = set_match.group(1)
            numbers = [parse_scientific_notation(
                x.strip()) for x in set_str.split(',')]

        target_factors = {}
        target_factors_match = re.search(
            r'TargetFactors\s*=\s*{([^}]+)}', content)
        if target_factors_match:
            factors_str = target_factors_match.group(1)
            factor_patterns = factors_str.split(',')

            for pattern in factor_patterns:
                pattern = pattern.strip()
                factor_match = re.search(r'(\d+)\^(\d+)', pattern)
                if factor_match:
                    prime = int(factor_match.group(1))
                    exponent = int(factor_match.group(2))
                    target_factors[prime] = exponent

            return {"numbers": numbers, "target": None, "target_factors": target_factors}

        target_match = re.search(
            r'Target\s*=\s*([0-9.e+-]+)', content, re.IGNORECASE)
        target = None
        if target_match:
            target_str = target_match.group(1)
            target = parse_scientific_notation(target_str)

        return {"numbers": numbers, "target": target, "target_factors": {}}

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return {"numbers": [], "target": None, "target_factors": {}}

# ==============================================================================
# Main Program Logic
# ==============================================================================


def print_diagnostics():
    """Print system capabilities and optimization availability"""
    print(f"=== Subset Product Solver ===")
    print(f"SymPy: {'Available' if HAS_SYMPY else 'Not available'}")
    print(f"GMPY2: {'Available' if HAS_GMPY2 else 'Not available'}")
    print(f"Parallel processing: {'Available' if HAS_MP else 'Not available'}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"===========================")


def main():
    """Main entry point for the program"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Subset Product Problem Solver')
    parser.add_argument('--data', default='data.txt',
                        help='Input data file (default: data.txt)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable factorization caching')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel factorization')
    args = parser.parse_args()

    initialize_cache()

    print_diagnostics()

    data = parse_input_file(args.data)
    if not data["numbers"]:
        print("Error: No numbers found in input file")
        return 1

    numbers = data["numbers"]
    target = data["target"]
    target_factors = data["target_factors"]

    # Setup for factorization
    use_cache = not args.no_cache
    use_parallel = not args.no_parallel

    if use_cache:
        print("Factorization caching: Enabled")
    else:
        print("Factorization caching: Disabled")

    if use_parallel and HAS_MP:
        print("Parallel factorization: Enabled")
    else:
        print("Parallel factorization: Disabled")

    # Create input_numbers dictionary
    input_numbers = {i: num for i, num in enumerate(numbers)}

    # Factorize all numbers
    numbers_to_factorize = list(input_numbers.items())
    factorization_results = factorize_number_list(
        numbers_to_factorize, use_parallel, use_cache)

    # Create number_factors dictionary
    number_factors = {i: factors for i, factors in factorization_results}

    # If we have a target without pre-computed factors, factorize it
    if target and not target_factors:
        print(f"Factorizing target: {target}")
        target_factors = factorize_large_number(target, use_cache)

    # Check for unfactorized numbers
    unfactorized = [i for i, factors in number_factors.items() if not factors]
    if unfactorized:
        print(f"Error: Failed to factorize numbers at indices: {unfactorized}")
        return 1

    if not target_factors:
        print("Error: No target factors found or computed")
        return 1

    # Generate and solve ASP program
    print("\nGenerating ASP program...")
    asp_program = create_asp_program(
        input_numbers, number_factors, target_factors)

    print("Solving subset product problem...")
    solution = solve_with_clingo(asp_program)

    # Parse and display solution
    result = parse_solution(solution)

    if result["selected_indices"]:
        print("\nSolution found!")
        print(f"Selected indices: {result['selected_indices']}")
        print(f"Selected values: {result['selected_values']}")

        product = 1
        for val in result["selected_values"]:
            product *= val

        if target:
            print(f"\nProduct of selected values: {product}")
            print(f"Target value: {target}")
            print(f"Match: {product == target}")
        else:
            print("\nProduct verified using prime factorization representation")
    else:
        print("\nNo solution found.")

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    exit_code = main()
    input("\nPress Enter to exit...")
    exit(exit_code)
    exit(exit_code)
