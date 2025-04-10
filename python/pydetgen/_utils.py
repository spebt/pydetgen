import sys, os


def get_valid_cvs_path(cvs_path: str):
    """
    Check if the csv file exists
    and return the path.
    """
    if not os.path.exists(cvs_path):
        print(f"The following csv file does not exist.\n  {cvs_path}")
        sys.exit(1)
    return cvs_path
