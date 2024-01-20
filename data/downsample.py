import csv
from sys import argv


def save_every_tenth_row(input_csv, output_csv):
    """writes a new CSV file containing each tenth row of the input CSV file"""

    with (open(input_csv, 'r') as infile,
          open(output_csv, 'w', newline='') as outfile):

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header to the output file
        header = next(reader)
        writer.writerow(header)

        # Write the first data row to the output file
        first_row = next(reader)
        writer.writerow(first_row)

        # Skip the next 8 rows (to get to the 10th row)
        # for _ in range(9):
        #    next(reader)

        # iterate through rows starting from the second, save every tenth
        for i, row in enumerate(reader, start=2):
            if (i - 1) % 10 == 0:
                writer.writerow(row)


if __name__ == "__main__":

    input_csv_file = argv[1]

    output_csv_file = argv[2]

    save_every_tenth_row(input_csv_file, output_csv_file)
