from pathlib import Path

source_path = "\\src"

current_dir = str(Path.cwd())
print(f"Current directory: {current_dir}")

if source_path in current_dir:
    dir = str(current_dir).split(source_path)[0]
    print("Source path found in current directory.")

PATHS = {
    "KIDNEY_DISEASE": f"{dir}\\datasets\\kidney-disease\\dataset.csv",
    "ADULT_INCOME": f"{dir}\\datasets\\adult-income\\adult.csv",
}
