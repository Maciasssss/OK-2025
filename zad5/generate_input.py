import random
import string
from pathlib import Path

MAX_LENGTH = 5 * 10**5  # 500,000 znaków
ALLOWED_CHARACTERS = string.ascii_lowercase  # Znaki od 'a' do 'z'
OUTPUT_FILENAME = "data.txt"


def create_max_data_file():
    """
    Tworzy plik data.txt z maksymalną dozwoloną liczbą znaków.
    """
    script_dir = Path(__file__).resolve().parent
    output_file_path = script_dir / OUTPUT_FILENAME

    print(
        f"Generowanie pliku '{output_file_path}' o długości {MAX_LENGTH} znaków...")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for _ in range(MAX_LENGTH):
                random_char = random.choice(ALLOWED_CHARACTERS)
                f.write(random_char)

        print(f"Pomyślnie wygenerowano plik '{output_file_path}'.")
        file_size = output_file_path.stat().st_size
        print(f"Rozmiar pliku: {file_size} bajtów.")

    except IOError as e:
        print(f"Błąd podczas zapisu do pliku '{output_file_path}': {e}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")


if __name__ == "__main__":
    create_max_data_file()
