#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code demonstrates how to use dedupe with a comma separated values
(CSV) file. All operations are performed in memory, so will run very
quickly on datasets up to ~10,000 rows.

We start with a CSV file containing our messy data. In this example,
it is listings of early childhood education centers in Chicago
compiled from several different sources.

The output will be a CSV with our clustered results.

For larger datasets, see our [mysql_example](mysql_example.html)
"""

import os
import sys
import csv
import re
import logging
import argparse
from argparse import RawTextHelpFormatter

import dedupe
from unidecode import unidecode

class IDmapper(dict):
	def __init__(self, n=0):
		self.n = n
	def get_id(self, key):
		if key not in self:
			self[key] = self.n
			self.n += 1
		return self[key]


def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def preProcess(column):
	"""
	Do a little bit of data cleaning with the help of Unidecode and Regex.
	Things like casing, extra spaces, quotes and new lines can be ignored.
	"""
	column = unidecode(column)
	column = re.sub('  +', ' ', column)
	column = re.sub('\n', ' ', column)
	column = column.strip().strip('"').strip("'").lower().strip()
	# If data is missing, indicate that by setting the value to `None`
	if not column:
		column = None
	return column


def readData(args):
	"""
	Read in our data from a CSV file and create a dictionary of records,
	where the key is a unique record ID and each value is dict
	"""
	data_d = {}
	with open(args.infile) as f:
		for line in f:
			cols = line.split('\t')
			row_id = args.idmapper.get_id(cols[0])
			data_d[row_id] = { 'id': cols[0], 'function':preProcess(cols[1]) }
	return data_d


if __name__ == '__main__':

	# ## Logging

	# Dedupe uses Python logging to show or suppress verbose output. This
	# code block lets you change the level of loggin on the command
	# line. You don't need it if you don't want that. To enable verbose
	# logging, run `python examples/csv_example/csv_example.py -v`
	usage = 'dedupe.py [-opt1, [-opt2, ...]] infile'
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file in fasta format')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write the output [stdout]')
	parser.add_argument('-v', '--verbose', action="store_true")
	parser.add_argument('--ids', action="store", help=argparse.SUPPRESS)
	args = parser.parse_args()

	'''
	(opts, args) = optp.parse_args()
	log_level = logging.WARNING
	if opts.verbose:
		if opts.verbose == 1:
			log_level = logging.INFO
		elif opts.verbose >= 2:
			log_level = logging.DEBUG
	logging.getLogger().setLevel(log_level)
	'''
	# ## Setup

	settings_file = args.infile + .'learned'
	training_file = args.infile + ".training"

	args.idmapper = IDmapper()
	
	print('importing data ...')
	data_d = readData(args)

	# If a settings file already exists, we'll just load that and skip training
	if os.path.exists(settings_file):
		print('reading from', settings_file)
		with open(settings_file, 'rb') as f:
			deduper = dedupe.StaticDedupe(f)
	else:
		# ## Training

		# Define the fields dedupe will pay attention to
		fields = [
			{'field': 'function', 'type': 'String'}
			]

		# Create a new deduper object and pass our data model to it.
		deduper = dedupe.Dedupe(fields)

		# If we have training data saved from a previous run of dedupe,
		# look for it and load it in.
		# __Note:__ if you want to train from scratch, delete the training_file
		if os.path.exists(training_file):
			print('reading labeled examples from ', training_file)
			with open(training_file, 'rb') as f:
				deduper.prepare_training(data_d, f)
		else:
			deduper.prepare_training(data_d)

		# ## Active learning
		# Dedupe will find the next pair of records
		# it is least certain about and ask you to label them as duplicates
		# or not.
		# use 'y', 'n' and 'u' keys to flag duplicates
		# press 'f' when you are finished
		print('starting active labeling...')

		dedupe.console_label(deduper)

		# Using the examples we just labeled, train the deduper and learn
		# blocking predicates
		deduper.train()

		# When finished, save our training to disk
		with open(training_file, 'w') as tf:
			deduper.write_training(tf)

		# Save our weights and predicates to disk.  If the settings file
		# exists, we will skip all the training and learning next time we run
		# this file.
		with open(settings_file, 'wb') as sf:
			deduper.write_settings(sf)

	# ## Clustering

	# `partition` will return sets of records that dedupe
	# believes are all referring to the same entity.

	print('clustering...')
	clustered_dupes = deduper.partition(data_d, 0.5)

	#print(clustered_dupes)
	print('# duplicate sets', len(clustered_dupes))

	# ## Writing Results

	# Write our original data back out to a CSV with a new column called
	# 'Cluster ID' which indicates which records refer to each other.

	cluster_membership = {}
	for cluster_id, (records, scores) in enumerate(clustered_dupes):
		for record_id, score in zip(records, scores):
			cluster_membership[record_id] = (cluster_id, score)

	print(cluster_membership)
	#print(cluster_membership)
	fieldnames = ['Cluster ID', 'confidence_score', 'id', 'function']
	with open(args.infile) as fp:
		for line in fp:
			cols = line.split('\t')
			row_id = args.idmapper.get_id(cols[0])
			membership = cluster_membership[row_id]
			args.outfile.write( str(membership[0]) )
			args.outfile.write( "\t" )
			args.outfile.write( str(membership[1])[:4] )
			args.outfile.write( "\t" )
			args.outfile.write(line)





