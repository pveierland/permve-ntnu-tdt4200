#!/usr/bin/python
import csv

def calculate_column_averages(filename):
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        values     = [ map(float, column[1:]) for column in zip(*csv_reader) ]
        return [ sum(column) / float(len(column)) for column in values ]

def find_and_store_averages_and_ratios(filename):
    averages = calculate_column_averages(filename)

    with open('averages_' + filename, 'w') as averages_file:
        averages_file.write(','.join(map(str, averages)))
        averages_file.write('\n')

    with open('total_ratios_' + filename, 'w') as total_ratios_file:
        total_ratios_file.write(','.join(map(str, (value / averages[1] for value in averages))))
        total_ratios_file.write('\n')
    
    with open('cuda_ratios_' + filename, 'w') as cuda_ratios_file:
        cuda_ratios_file.write(','.join(map(str, (value / (averages[1] - averages[2] - averages[3] - averages[4]) for value in averages))))
        cuda_ratios_file.write('\n')

find_and_store_averages_and_ratios('lenna_20_32.txt')
find_and_store_averages_and_ratios('lenna_384_256.txt')

