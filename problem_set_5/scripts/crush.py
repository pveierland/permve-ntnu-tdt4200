#!/usr/bin/python

import random
import subprocess
import sys

class colors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

def run_program(program, inputs):
    proc   = subprocess.Popen([program], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    result = proc.communicate(
        '{0}\n{1}'.format(
            len(inputs),
            '\n'.join('{0} {1}'.format(*i) for i in inputs)).encode('utf-8'))
    if not proc.returncode:
        return map(int, result[0].decode('utf-8').splitlines())

def verify_test_set(start, stop, threads, expected):
    outputs = run_program(sys.argv[1], [ (start, stop) ])
    success = outputs == expected

    print('{0}[{1}] start={2} stop={3} threads={4} expected={5} actual={6}{7}'.format(
        colors.OKGREEN if success else colors.FAIL,
        'OK' if success else 'FAIL',
        start, stop, threads, expected, outputs, colors.ENDC))

    if not success:
        sys.exit(1)

# Valid c values + number of triplets for each c:
valid_c_values = [
    (5, 1),
    (13, 1),
    (17, 1),
    (25, 1),
    (29, 1),
    (37, 1),
    (41, 1),
    (53, 1),
    (61, 1),
    (65, 2),
    (73, 1),
    (85, 2),
    (89, 1),
    (97, 1)
]

for start in range(-5, 101):
    for stop in range(-5, 101):
        expected = sum(c[1] for c in valid_c_values if c[0] >= start and c[0] < stop) if (start >= 0 and stop >= 0) else 0
        verify_test_set(start, stop, random.randrange(1, 9), [expected])
