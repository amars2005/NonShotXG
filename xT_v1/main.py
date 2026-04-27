# main.py
import os
from builder import DatasetBuilder

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    builder = DatasetBuilder()

    print("Starting data extraction...")
    raw_data = builder.build_dataset()

    if raw_data.empty:
        print("No data found. Exiting.")
        return

    final_data = builder.process_chains(raw_data)

    output_filename = os.path.join(_ROOT, "statsbomb_chained_dataset.csv")
    final_data.to_csv(output_filename, index=False)

    print("\n--- DONE ---")
    print(f"Data saved to {output_filename}")
    print(f"Total Rows: {len(final_data)}")
    print(f"Columns: {list(final_data.columns)}")

if __name__ == "__main__":
    main()