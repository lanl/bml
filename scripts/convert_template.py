#!/usr/bin/env python

re_D = {}


def init_conversion(defs):
    """Initialize the regular expressions given a list of definitions.
    """

    import re

    for definition in defs:
        tokens = definition.split("=")
        if len(tokens) != 2:
            continue
        re_D[re.compile(tokens[0].strip())] = tokens[1].strip()


def convert_line(line, defs):
    """Convert a line given a list of definitions.

    The definitions are in the standard preprocessor form, i.e. -DA or
    -DA=VALUE.
    """

    new_line = line.rstrip()
    for D in re_D:
        new_line = D.sub(re_D[D], new_line)
    return new_line


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="This script acts in a very " +
        "similar way as the C preprocessor would. " +
        "It's just a lot simpler. :)")

    parser.add_argument("SOURCE", help="The Fortran source file")

    parser.add_argument("-o",
                        metavar="OUTPUT",
                        help="Write output to file OUTPUT")

    parser.add_argument("-D", help="Preprocessor macro", action="append")

    options = parser.parse_args()

    init_conversion(options.D)

    import sys
    if options.o is None:
        fd_out = sys.stdout
    else:
        fd_out = open(options.o, "w")

    fd_in = open(options.SOURCE)
    for line in fd_in:
        fd_out.write(convert_line(line, options.D) + "\n")
    fd_out.close()
    fd_in.close()
