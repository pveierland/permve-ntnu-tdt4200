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

def verify_test_set(program, inputs, expected):
    outputs = run_program(program, inputs)
    success = outputs == expected

    print('{0}[{1}] {2} inputs={3} expected={4} actual={5}{6}'.format(
        colors.OKGREEN if success else colors.FAIL,
        'OK' if success else 'FAIL',
        program, inputs, expected, outputs, colors.ENDC))

    if not success:
        sys.exit(1)

test_inputs      = [ [ (0, 100), (75, 150), (150, 75) ] ]
expected_outputs = [ [ 16, 12, 0 ] ]

for mpi_nodes in range(1, 11):
    for openmp_threads in range(1, 11):
        for test_input, expected_output in zip(test_inputs, expected_outputs):
            verify_test_set(
                'mpirun -n {0} {1}'.format(mpi_nodes, sys.argv[1]),
                list((start, stop, openmp_threads) for start, stop in test_input),
                expected_output)
