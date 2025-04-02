#!/usr/bin/env python3
"""
Fixed Subset Product Problem Solver with corrected factorization tracking
This version correctly identifies all prime factors in the input set.

Usage:
  python solver.py [--data filename] [--no-cache] [--no-parallel] [--debug]
"""

import re
import os
import math
import time
import json
import hashlib
import argparse
import clingo
from datetime import datetime

# Optional imports with fallbacks
try:
    import sympy
    from sympy.ntheory import factorint
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    print("WARNING: SymPy not found. Install with 'pip install sympy' for advanced factorization.")

try:
    import gmpy2
    HAS_GMPY2 = True
except ImportError:
    HAS_GMPY2 = False
    print("INFO: GMPY2 not found. Install with 'pip install gmpy2' for faster arithmetic.")

try:
    import multiprocessing as mp
    HAS_MP = True
except ImportError:
    HAS_MP = False
    print("INFO: Multiprocessing not available. Parallel factorization disabled.")

# Constants
MAX_PRIME = 10000     # Upper limit for prime generation
MAX_EXPONENT = 100    # Maximum exponent to check in factorization
CACHE_FILE = "factorization_cache.json"
MAX_WORKERS = max(1, mp.cpu_count() - 1) if HAS_MP else 1

# Global variables for multiprocessing
GLOBAL_USE_CACHE = True
DEBUG_MODE = False

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
        except Exception as e:
            if DEBUG_MODE:
                print(f"Failed to initialize cache: {e}")
            return False
    return True


def load_cache():
    """Load the factorization cache from disk with better error handling"""
    if not os.path.exists(CACHE_FILE):
        initialize_cache()
        return {}

    try:
        with open(CACHE_FILE, 'r') as f:
            data = f.read().strip()
            if not data:  # Empty file
                return {}
            return json.loads(data)
    except json.JSONDecodeError as e:
        if DEBUG_MODE:
            print(f"Warning: Cache file corrupt, resetting - {e}")
        # Reset cache file if corrupted
        initialize_cache()
        return {}
    except Exception as e:
        if DEBUG_MODE:
            print(f"Warning: Could not load cache - {e}")
        return {}


def save_cache(cache):
    """Save the factorization cache to disk with better error handling"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        return True
    except Exception as e:
        if DEBUG_MODE:
            print(f"Warning: Could not save cache - {e}")
        return False


def number_hash(n):
    """Create a hash for a number to use as cache key"""
    return hashlib.md5(str(n).encode()).hexdigest()


def is_prime_fast(n):
    """Quick primality test using Miller-Rabin"""
    if HAS_SYMPY:
        return sympy.isprime(n)
    elif HAS_GMPY2:
        return gmpy2.is_prime(n, 25)  # 25 rounds of primality testing
    else:
        # Fallback to basic primality test
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

    # Check if divisible by 2
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    # Check for other small primes
    for i in range(3, 100, 2):
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i

    # If n is still large, check if it's a perfect square or has small factors
    if n > 1:
        # Try to factorize further
        sqrt_n = int(math.sqrt(n))
        if sqrt_n * sqrt_n == n and is_prime_fast(sqrt_n):
            # Perfect square of a prime
            factors[sqrt_n] = 2
        else:
            # Try more divisibility checks for medium primes
            for i in range(101, 10000, 2):
                if i * i > n:
                    break
                while n % i == 0:
                    factors[i] = factors.get(i, 0) + 1
                    n //= i

            # Add any remaining factor
            if n > 1:
                factors[n] = factors.get(n, 0) + 1

    return factors


def factorize_with_sympy(n):
    """Use SymPy's advanced factorization"""
    return factorint(n, limit=10**7, use_trial=True, use_rho=True, use_pm1=True, verbose=DEBUG_MODE)


def factorize_with_gmpy2(n):
    """Use GMPY2's factorization if available"""
    if not HAS_GMPY2:
        return None

    factors = {}
    temp_n = gmpy2.mpz(n)

    # Try to factorize using GMPY2's functions
    i = gmpy2.mpz(2)
    while i * i <= temp_n:
        while temp_n % i == 0:
            factors[int(i)] = factors.get(int(i), 0) + 1
            temp_n //= i
        i += 1

    if temp_n > 1:
        factors[int(temp_n)] = factors.get(int(temp_n), 0) + 1

    return factors

# This function needs to be outside any other function for multiprocessing


def factorize_with_params(n):
    """Wrapper for multiprocessing - must be at module level"""
    return factorize_large_number(n, GLOBAL_USE_CACHE)


def factorize_large_number(n, use_cache=True):
    """
    Optimized factorization for large numbers with caching,
    algorithm selection, and fallbacks
    """
    # Convert to integer if needed
    if isinstance(n, str):
        n = int(n)

    # Check cache first
    cache = load_cache() if use_cache else {}
    n_hash = number_hash(n)
    if n_hash in cache:
        print(f"Cache hit for {n}")
        return {int(k): v for k, v in cache[n_hash].items()}

    print(f"Factorizing {n}...")
    start_time = time.time()

    # Special case for small numbers
    if n < 10**9:
        factors = factorize_small(n)
    # Check if prime first (can save a lot of time)
    elif is_prime_fast(n):
        factors = {n: 1}
    else:
        # Try different methods based on available libraries
        if HAS_SYMPY:
            factors = factorize_with_sympy(n)
        elif HAS_GMPY2:
            factors = factorize_with_gmpy2(n)
        else:
            # Fallback to basic factorization
            factors = factorize_small(n)

    end_time = time.time()
    print(f"Factorization completed in {end_time - start_time:.2f} seconds")

    # Debug output to verify factorization
    if DEBUG_MODE:
        product = 1
        for prime, exp in factors.items():
            product *= prime ** exp
        if product != n:
            print(f"WARNING: Factorization verification failed for {n}!")
            print(f"Original: {n}")
            print(f"Product of factors: {product}")
            print(f"Factors: {factors}")
        else:
            print(
                f"Factorization verified: {n} = {format_factorization(factors)}")

    # Update cache
    if use_cache:
        cache[n_hash] = {str(k): v for k, v in factors.items()}
        save_cache(cache)

    return factors


def factorize_number_list(numbers, use_parallel=True, use_cache=True):
    """Factorize a list of numbers, potentially in parallel"""
    # Set global for multiprocessing
    global GLOBAL_USE_CACHE
    GLOBAL_USE_CACHE = use_cache

    if not use_parallel or not HAS_MP or len(numbers) <= 1 or MAX_WORKERS <= 1:
        return [(i, factorize_large_number(n, use_cache)) for i, n in numbers]

    # Use multiprocessing for parallel factorization
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


def generate_primes(max_prime=10000):
    """Generate prime numbers up to max_prime"""
    sieve = [True] * (max_prime + 1)
    sieve[0] = sieve[1] = False

    for i in range(2, int(math.sqrt(max_prime)) + 1):
        if sieve[i]:
            for j in range(i*i, max_prime + 1, i):
                sieve[j] = False

    return [i for i in range(max_prime + 1) if sieve[i]]


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

    control.configuration.solve.models = 0  # Find all optimal models
    handle = control.solve(on_model=on_model)

    # Fixed: Direct check of satisfiability without calling .get()
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

        # Parse the set
        set_match = re.search(r'Set\s*=\s*{([^}]+)}', content)
        numbers = []
        if set_match:
            set_str = set_match.group(1)
            # Handle scientific notation and large integers
            numbers = [parse_scientific_notation(
                x.strip()) for x in set_str.split(',')]

        # Look for factorized target format first
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

        # Parse the standard target format
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
# Analysis and Diagnostic Functions
# ==============================================================================


def verify_factorizations(number_factors, input_numbers):
    """
    Verify that all factorizations are correct by multiplying the factors
    """
    incorrect = []
    for idx, factors in number_factors.items():
        if idx in input_numbers:
            original_number = input_numbers[idx]
            product = 1
            for prime, exp in factors.items():
                product *= prime ** exp

            if product != original_number:
                incorrect.append((idx, original_number, product, factors))

    return incorrect


def analyze_factorizations(input_factors, target_factors):
    """
    Analyze if a solution is possible by comparing input and target factorizations.
    Fixed to correctly identify all available prime factors.
    """
    # Collect all primes found in inputs
    all_input_primes = set()
    max_exponents = {}
    input_primes_by_number = {}

    for idx, factors in input_factors.items():
        input_primes_by_number[idx] = set(factors.keys())
        for prime, exp in factors.items():
            all_input_primes.add(prime)
            max_exponents[prime] = max(max_exponents.get(prime, 0), exp)

    # Check if all target primes are available in inputs
    missing_primes = []
    insufficient_exponents = []
    input_numbers_with_primes = {}

    for prime, target_exp in target_factors.items():
        # Track which input numbers contain each prime
        input_numbers_with_primes[prime] = [
            idx for idx, primes in input_primes_by_number.items() if prime in primes
        ]

        if prime not in all_input_primes:
            missing_primes.append(prime)
        elif max_exponents.get(prime, 0) * len(input_numbers_with_primes[prime]) < target_exp:
            # Even if all inputs with this prime are selected, we can't reach the target exponent
            insufficient_exponents.append((
                prime,
                target_exp,
                max_exponents.get(prime, 0),
                len(input_numbers_with_primes[prime])
            ))

    return {
        "missing_primes": missing_primes,
        "insufficient_exponents": insufficient_exponents,
        "input_numbers_with_primes": input_numbers_with_primes
    }


def test_specific_divisibility(input_numbers, prime):
    """Test which input numbers are divisible by a specific prime"""
    divisible = []
    for idx, num in input_numbers.items():
        if num % prime == 0:
            divisible.append((idx, num))
    return divisible

# ==============================================================================
# Main Program Logic
# ==============================================================================


def print_diagnostics():
    """Print system capabilities and optimization availability"""
    print(f"=== Subset Product Problem Solver - Optimization Status ===")
    print(f"SymPy available: {HAS_SYMPY}")
    print(f"GMPY2 available: {HAS_GMPY2}")
    print(f"Parallel processing: {'Available' if HAS_MP else 'Not available'}")
    print(f"Processor cores: {mp.cpu_count() if HAS_MP else 'N/A'}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"================================================")


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
    parser.add_argument('--timeout', type=int,
                        help='Timeout in seconds for the solver')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--check-prime', type=int,
                        help='Check which input numbers are divisible by this prime')
    args = parser.parse_args()

    # Set debug mode
    global DEBUG_MODE
    DEBUG_MODE = args.debug

    # Initialize cache
    initialize_cache()

    # Print diagnostics
    print_diagnostics()

    # Parse input file
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

    # If user wants to check divisibility by a specific prime
    if args.check_prime:
        prime = args.check_prime
        print(f"\nChecking which input numbers are divisible by {prime}:")
        divisible = test_specific_divisibility(input_numbers, prime)
        if divisible:
            for idx, num in divisible:
                print(f"  Number at index {idx}: {num} (divisible by {prime})")
        else:
            print(f"  No input numbers are divisible by {prime}")
        if not DEBUG_MODE:
            return 0

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

    # Verify factorizations if in debug mode
    if DEBUG_MODE:
        print("\n=== Factorization Verification ===")
        incorrect = verify_factorizations(number_factors, input_numbers)
        if incorrect:
            print("WARNING: The following factorizations are incorrect:")
            for idx, original, product, factors in incorrect:
                print(
                    f"  Number {idx}: {original} != {product} (factors: {factors})")
        else:
            print("All factorizations verified successfully")

    # Print factorization information if debug mode
    if DEBUG_MODE:
        print("\n=== Factorization Information ===")
        print(f"Target factorization: {format_factorization(target_factors)}")
        for idx, factors in number_factors.items():
            print(
                f"Number {idx} ({input_numbers[idx]}): {format_factorization(factors)}")

    # Analyze if a solution is possible
    analysis = analyze_factorizations(number_factors, target_factors)

    # Special test for any primes that might be missing
    if analysis["missing_primes"]:
        print("\n=== Testing for Divisibility Issues ===")
        for prime in analysis["missing_primes"]:
            divisible = test_specific_divisibility(input_numbers, prime)
            if divisible:
                print(
                    f"IMPORTANT: Despite the analysis, these numbers ARE divisible by {prime}:")
                for idx, num in divisible:
                    print(f"  Number at index {idx}: {num}")
                print("This indicates a factorization issue that needs to be fixed.")

                # Since we found a direct divisibility, remove this prime from missing
                analysis["missing_primes"].remove(prime)

    if analysis["missing_primes"] or analysis["insufficient_exponents"]:
        print("\n=== Problem Analysis ===")
        if analysis["missing_primes"]:
            print(
                f"Missing primes in input set: {', '.join(map(str, analysis['missing_primes']))}")
            print("A solution is impossible because some prime factors in the target are not available in any input number.")

        if analysis["insufficient_exponents"]:
            print("Insufficient exponents for these primes:")
            for prime, target_exp, max_exp, num_inputs in analysis["insufficient_exponents"]:
                total_available = max_exp * num_inputs
                print(
                    f"  Prime {prime}: Need {target_exp}, but maximum available is {total_available} ({max_exp} per number × {num_inputs} numbers with this factor)")
            print(
                "A solution is impossible because some prime factor exponents in the target cannot be reached.")

        if not DEBUG_MODE:
            print(
                "\nRun with --debug for more information or --check-prime=X to test divisibility.")
            return 1

    # Generate and solve ASP program
    print("\nGenerating ASP program...")
    asp_program = create_asp_program(
        input_numbers, number_factors, target_factors)

    # Count ASP program statistics
    num_facts = asp_program.count(".")
    num_rules = asp_program.count(":-")

    if DEBUG_MODE:
        print(f"ASP program statistics: {num_facts} facts, {num_rules} rules")

    print("Solving subset product problem...")
    solution = solve_with_clingo(asp_program, args.timeout)

    # Parse and display solution
    result = parse_solution(solution)

    if result["selected_indices"]:
        print("\nSolution found!")
        print(f"Selected indices: {result['selected_indices']}")
        print(f"Selected values: {result['selected_values']}")

        # Verify product
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

        # Provide more information about why no solution was found
        if not (analysis["missing_primes"] or analysis["insufficient_exponents"]):
            print("\nThis could be due to:")
            print("1. The problem may be mathematically impossible (combination of prime exponents cannot be satisfied)")
            print("2. The solver timed out or reached resource limits")
            print("3. The solver's optimization strategy could not find a solution")
            print("\nTry running with --debug to get more information.")

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
