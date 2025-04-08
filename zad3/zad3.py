import sys
import time
from typing import List, Tuple


def maksymalna_liczba_stron(budzet: int, ceny: List[int], strony: List[int]) -> int:
    dp = [0] * (budzet + 1)
    price_zero_pages = 0
    valid_items = []

    # Use a set to track unique (price, pages) pairs
    unique_books = set()

    # Preprocess to separate price 0 items and filter out items exceeding the budget
    for price, pages in zip(ceny, strony):
        if (price, pages) in unique_books:
            continue  # Skip duplicate books
        unique_books.add((price, pages))

        if price == 0:
            price_zero_pages += pages
        elif price <= budzet:
            valid_items.append((price, pages))

    # Apply price 0 pages to all positions once
    if price_zero_pages > 0:
        for j in range(budzet + 1):
            dp[j] += price_zero_pages

    # Process valid items
    dp_local = dp
    for price, pages in valid_items:
        for j in range(budzet, price - 1, -1):
            prev = j - price
            new_val = dp_local[prev] + pages
            if new_val > dp_local[j]:
                dp_local[j] = new_val

    return dp_local[budzet]


def parse_input(input_data: str) -> Tuple[List[int], List[int]]:
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
        with open('data.txt', 'r') as file:
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
        print("\nError: data.txt file not found!")
        print("Ensure data.txt is in the same directory as the script.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

    print("\nProgram completed.")


if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    exit(exit_code)
