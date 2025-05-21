import sys
import time
import json
from typing import List, Tuple
from math import gcd as math_gcd


def maksymalna_liczba_stron(budzet: int, ceny: List[int], strony: List[int]) -> int:
    n = len(ceny)
    if n == 0 or budzet <= 0:
        return 0

    # Fast path: If we can buy all books, return total pages
    total_cost = 0
    for cost in ceny:
        total_cost += cost
    if total_cost <= budzet:
        return sum(strony)

    # Extract zero-cost books (always include them)
    zero_cost_pages = 0
    non_zero_costs = []
    non_zero_pages = []

    for i in range(n):
        if ceny[i] == 0:
            zero_cost_pages += strony[i]
        else:
            non_zero_costs.append(ceny[i])
            non_zero_pages.append(strony[i])

    # No non-zero cost books? Return just the zero-cost pages
    if not non_zero_costs:
        return zero_cost_pages

    # Reduce the problem size by finding the GCD of all costs
    current_gcd = non_zero_costs[0]
    for cost in non_zero_costs[1:]:
        current_gcd = math_gcd(current_gcd, cost)

    # Scale down the costs and budget if possible
    if current_gcd > 1:
        scaled_costs = [c // current_gcd for c in non_zero_costs]
        scaled_budget = budzet // current_gcd
    else:
        scaled_costs = non_zero_costs
        scaled_budget = budzet

    # Initialize dp array with smallest possible size
    dp = [0] * (scaled_budget + 1)

    # Perform the knapsack DP algorithm with optimizations
    for i in range(len(scaled_costs)):
        cost = scaled_costs[i]
        page = non_zero_pages[i]

        if page <= 0:
            continue

        # Use faster reverse iteration to avoid duplicate counting
        # Only iterate through valid budget ranges
        for j in range(scaled_budget, cost - 1, -1):
            new_value = dp[j - cost] + page
            if new_value > dp[j]:
                dp[j] = new_value

    # Return the result including zero-cost book pages
    return dp[scaled_budget] + zero_cost_pages


def parse_input(input_data: str) -> Tuple[List[int], List[int]]:
    """Parse JSON input data into lists of costs and pages."""
    ceny = []
    strony = []

    try:
        books = json.loads(input_data)
        for book in books:
            ceny.append(book["cost"])
            strony.append(book["pages"])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: Invalid JSON format or missing keys: {e}")
        sys.exit(1)

    return ceny, strony


def main():
    print("=" * 60)
    print("BOOK PAGE MAXIMIZATION PROBLEM")
    print("=" * 60)

    try:
        with open('books.json', 'r') as file:
            input_data = file.read()

        ceny, strony = parse_input(input_data)

        while True:
            try:
                budzet = int(input("\nEnter available budget (x) in PLN: "))
                if 1 <= budzet <= 10**5:
                    break
                print(f"Error: Budget must be between 1 and {10**5}.")
            except ValueError:
                print("Error: Please enter a valid integer.")

        start_time = time.time()
        wynik = maksymalna_liczba_stron(budzet, ceny, strony)
        end_time = time.time()

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"\nMaximum number of pages that can be purchased: {wynik}")

        print("\nDetails:")
        print(f"- Number of available books: {len(ceny)}")
        print(f"- Available budget: {budzet} PLN")

        execution_time = end_time - start_time
        print(f"- Execution time: {execution_time:.4f} seconds")

        if execution_time > 10:
            print("\nWARNING: Execution time exceeded 10 seconds!")
        else:
            print(
                f"\nExecution time within 10-second limit (used {execution_time/10*100:.1f}% of limit)")

    except FileNotFoundError:
        print("\nError: books.json file not found!")
        print("Ensure books.json is in the same directory as the script.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

    print("\nProgram completed.")


if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
