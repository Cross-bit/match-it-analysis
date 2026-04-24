import subprocess
from utils.config import THESIS_PROJECT_ROOT
import argparse
import shutil
import os

def get_real_file_path(tex_file_name):
    return os.path.join(THESIS_PROJECT_ROOT, tex_file_name)

def make_backup(latex_files, backup_dir = "backup"):
    """Creates backups of the files updated."""
    abs_path = os.path.join(os.path.dirname(__file__), backup_dir)

    if (os.path.exists(abs_path)):
        shutil.rmtree(abs_path)

    os.mkdir(abs_path)

    failed_backup = []
    for tex_file in latex_files:

        file_path = get_real_file_path(tex_file)
        if os.path.isfile(file_path):
            shutil.copy(file_path, abs_path)
        else:
            print(f"Skipping {tex_file} — not a file")
            failed_backup.append(tex_file)

    return failed_backup


def main():


    latex_files = ["thesis.tex"] # list of files to update tables in

    print("Making backups...")
    failed_backups = make_backup(latex_files)
    if (failed_backups != []):
        print(failed_backups)
        exit(1)

    print("Starting orchestration...")

    subprocess.run(["python3", "-m", "restaurant_data.easer_experiments.optimal_easer_lambda"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple automatic table update script")
    parser.add_argument("--labels", type=str, help="List of table label names to update in the files.")
    args = parser.parse_args()

    main()

