import csv


def save_every_tenth_row(input_csv, output_csv):

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

    # Replace with the path to your input CSV file
    input_csv_file = 'data/training2_simple_100Hz.csv'

    # Replace with the desired output CSV file
    output_csv_file = 'data/training2_simple_10Hz.csv'

    save_every_tenth_row(input_csv_file, output_csv_file)
