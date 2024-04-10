import sys

def combine_files(files : list[str], outfile: str):
    with open(outfile, 'w') as out:
        output = ""
        for file in files:
            print(file)
            with open(file) as f:
                output += f.read() + '\n'

        print(f'writing to {outfile}')
        out.write(output)


if __name__ == '__main__':
    input_files = sys.argv[1:-1]
    output_file = sys.argv[-1]

    combine_files(input_files, output_file)
