import sys
import time
from typing import List, Tuple


def maksymalna_liczba_stron(budzet: int, ceny: List[int], strony: List[int]) -> int:
    """
    Solve the 0/1 Knapsack problem with all input books.

    Args:
        budzet (int): Available budget
        ceny (List[int]): List of book prices
        strony (List[int]): List of book page counts

    Returns:
        int: Maximum number of pages that can be purchased
    """
    # Initialize DP array
    dp = [0] * (budzet + 1)

    # Iterate through all books
    for price, pages in zip(ceny, strony):
        # Iterate in reverse to prevent multiple uses of same book
        for j in range(budzet, price - 1, -1):
            # Update maximum pages for current budget
            dp[j] = max(dp[j], dp[j - price] + pages)

    return dp[budzet]


def parse_input(input_data: str) -> Tuple[List[int], List[int]]:
    """
    Parse input data efficiently.

    Args:
        input_data (str): Raw input string from file

    Returns:
        Tuple of (prices list, pages list)
    """
    ceny = []
    strony = []

    for line in input_data.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        try:
            price, pages = map(int, line.split())
            ceny.append(price)
            strony.append(pages)
        except ValueError:
            print(f"Warning: Invalid input line: {line}")

    return ceny, strony


def main():
    print("=" * 60)
    print("BOOK PAGE MAXIMIZATION PROBLEM")
    print("=" * 60)

    try:
        # Read entire file at once
        with open('data.txt', 'r') as file:
            input_data = file.read()

        # Parse input data
        ceny, strony = parse_input(input_data)

        # Get budget
        while True:
            try:
                budzet = int(input("\nEnter available budget (x) in PLN: "))
                if 1 <= budzet <= 10**5:
                    break
                print(f"Error: Budget must be between 1 and {10**5}.")
            except ValueError:
                print("Error: Please enter a valid integer.")

        # Solve the problem
        start_time = time.time()
        wynik = maksymalna_liczba_stron(budzet, ceny, strony)
        end_time = time.time()

        # Print results
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
        print("\nError: data.txt file not found!")
        print("Ensure data.txt is in the same directory as the script.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

    print("\nProgram completed.")


if __name__ == "__main__":
    main()
