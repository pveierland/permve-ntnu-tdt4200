#!/usr/bin/python

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
    proc   = subprocess.Popen([program], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = proc.communicate(
        '{0}\n{1}'.format(
            len(inputs),
            '\n'.join('{0} {1}'.format(*i) for i in inputs)).encode('utf-8'))
    if not proc.returncode:
        return map(int, result[0].decode('utf-8').splitlines())

def verify_test_set(start, stop, expected):
    outputs = run_program(sys.argv[1], [ (start, stop) ])
    success = outputs == [expected]

    print('{0}[{1}] start={2} stop={3} expected={4} actual={5}{6}'.format(
        colors.OKGREEN if success else colors.FAIL,
        'OK' if success else 'FAIL',
        start, stop, expected, outputs, colors.ENDC))

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

for start in range(0, 100):
    for stop in range(0, 100):
        expected = sum(c[1] for c in valid_c_values if c[0] >= start and c[0] < stop)
        verify_test_set(start, stop, expected)
