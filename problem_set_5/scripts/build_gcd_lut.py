#!/usr/bin/python3

import argparse
import fractions

parser = argparse.ArgumentParser()
parser.add_argument('M', type=int)
parser.add_argument('N', type=int)
args = parser.parse_args()

print('static const unsigned char coprime_lut[{0}][{1}] = {{'.format(
    args.M, args.N))

for m in range(1, args.M + 1):
    print('{ ', end='')
    for n in range(1, args.N + 1):
        print('{0}{1} '.format(
            '1' if fractions.gcd(m, n) == 1 else '0',
            ',' if n != args.N else ''), end='')
    print('}}{0}'.format(',' if m != args.M else ''))

print('};')

