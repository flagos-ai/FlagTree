#!/usr/bin/env python3
"""Compare two folders and output differences to a file."""

import argparse
import difflib
import os
import sys


def get_relative_paths(folder):
    result = set()
    for root, _, files in os.walk(folder):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, folder)
            result.add(rel)
    return result


def is_binary(filepath):
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(8192)
            return b'\x00' in chunk
    except IOError:
        return True


def read_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()
    except IOError:
        return []


def compare_folders(folder_base, folder_pr, output_dir):
    base_paths = get_relative_paths(folder_base)
    pr_paths = get_relative_paths(folder_pr)

    only_in_base = sorted(base_paths - pr_paths)
    only_in_pr = sorted(pr_paths - base_paths)
    common = sorted(base_paths & pr_paths)

    diff_files = []
    skipped_files = []
    unchanged_mlir_files = []
    same_count = 0

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for rel in common:
        # compare .mlir files only
        if not rel.endswith('.mlir'):
            print(f" Not a mlir file: {rel}, skipped")
            continue

        full_name_base = os.path.join(folder_base, rel)
        full_name_pr = os.path.join(folder_pr, rel)
        if is_binary(full_name_base) or is_binary(full_name_pr):
            # skip binary file
            print(f" Encountered binary file, skipped: {rel}")
            skipped_files.append(rel)
            continue
        else:
            try:
                # Text files can be read into lines for difflib
                lines_base = read_lines(full_name_base)
                lines_pr = read_lines(full_name_pr)

                if lines_base != lines_pr:
                    diff_files.append(rel)
                    diff = difflib.unified_diff(lines_base, lines_pr, fromfile=f'base/{rel}', tofile=f'pr/{rel}',
                                                lineterm='')
                    diff_text = ''.join(diff)
                    if diff_text:
                        _write_diff_file(rel, diff_text + "\n", output_dir)
                else:
                    same_count += 1
                    unchanged_mlir_files.append(rel)
            except IOError:
                diff_files.append(rel)
                _write_diff_file(rel, f"  Cannot read: {rel}\n", output_dir)

    # names of unchanged .mlir files will be written into unchanged_cases.txt
    if unchanged_mlir_files:
        changed_txt_path = os.path.join(output_dir, "unchanged_cases.txt")
        print(f"\nUnchanged mlir files written to : '{changed_txt_path}/'")
        with open(changed_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unchanged_mlir_files) + '\n')

    # Print summary to console
    print(f"\n{'='*40}")
    print(f"Comparison Summary")
    print(f"{'='*40}")
    print(f"  Only in base count: {len(only_in_base)}")
    print(f"  Only in pr count: {len(only_in_pr)}")
    print(f"  Identical count: {same_count}")
    print(f"  Different count: {len(diff_files)}")
    print(f"  Skipped (binary) count: {len(skipped_files)}")

    if diff_files:
        print(f"\nDifferences written to directory: '{output_dir}/'")


def _write_diff_file(rel_path, content, output_dir):
    # Helper to write diff content to a specific file based on its base name.
    base_name = os.path.basename(rel_path)
    diff_filename = os.path.splitext(base_name)[0] + '.diff'
    diff_path = os.path.join(output_dir, diff_filename)

    try:
        with open(diff_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Writing diff file {diff_filename} failed, exception is : {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare base and base+pr mlir folders and output differences to a file.')
    parser.add_argument('folder_base', help='Path to base folder')
    parser.add_argument('folder_pr', help='Path to pr folder')
    parser.add_argument('output',
                        help='Path to output directory where diff files and unchanged_cases.txt will be stored')
    args = parser.parse_args()

    if not os.path.isdir(args.folder_base):
        print(f"Error: '{args.folder_base}' is not a directory", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.folder_pr):
        print(f"Error: '{args.folder_pr}' is not a directory", file=sys.stderr)
        sys.exit(1)

    compare_folders(args.folder_base, args.folder_pr, args.output)


if __name__ == '__main__':
    main()
