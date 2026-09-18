"""Decode the timestamp that swin_train.py generates and adds to filenames."""

import argparse
from datetime import datetime, timezone

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Decode the hex timestamp that appears in filenames.')
    required = parser.add_argument_group('required arguments')

    required.add_argument('-hex', '--hex_value', required=True, type=str,
                          help='hex value to decode')

    args = parser.parse_args()
    return args

def main():
    """Main function."""
    args = parse_arguments()
    timestamp_int = int(args.hex_value, 16)

    # Convert the integer timestamp into a datetime object
    date_time_obj = datetime.fromtimestamp(timestamp_int)
    print(f"Date and Time: {date_time_obj}")


if __name__ == "__main__":
    main()
