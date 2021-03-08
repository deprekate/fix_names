#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code demonstrates how to use dedupe with a comma separated values
(CSV) file. All operations are performed in memory, so will run very
quickly on datasets up to ~10,000 rows.

"""

import os
import sys
import csv
import re
import logging
import argparse
from argparse import RawTextHelpFormatter
from io import StringIO

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
	column = unidecode(column).lower()
	# gene function related regex
	if 'hypothetical' in column or 'unknown' in column:
		column = 'hypothetical'
	# TO DROP
	column = column.replace('similar to ','')
	column = column.replace('bacteriophage-','')
	column = column.replace('bacteriophage','')
	column = column.replace('phage-','')
	column = column.replace('phage','')
	column = column.replace('associated','')
	column = column.replace('aquired','')
	column = column.replace('domain containing','')
	column = column.replace('domain-containing','')
	column = column.replace('domain','')
	column = column.replace('conserved','')
	column = column.replace('predicted','')
	column = column.replace('putative','')
	column = column.replace('protein','')
	column = column.replace('homolog','')
	column = column.replace('analog','')
	column = column.replace('-like','')
	column = column.replace(' like','')
	# TO REPLACE
	column = column.replace('base plate','baseplate')

	# OTHER
	column = re.sub(r'\bgp\d+', ' ', column)
	column = re.sub(r'\borf\d+', ' ', column)
	column = re.sub(r'\borf \d+', ' ', column)

	column = re.sub('  +', ' ', column)
	column = re.sub('\n', ' ', column)
	column = column.strip().strip('"').strip("'").strip()
	# If data is missing, indicate that by setting the value to `None`
	if not column:
		column = None
	return column


def readData(args):
	"""
	Read in our data
	"""
	data_d = {}
	with open(args.infile) as f:
		for line in f:
			cols = line.split('\t')
			row_id = args.idmapper.get_id(cols[0])
			data_d[row_id] = { 'id': cols[0], 'function':preProcess(cols[1]) }
	return data_d


if __name__ == '__main__':

	usage = 'dedupe.py [-opt1, [-opt2, ...]] infile'
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file in fasta format')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write the output [stdout]')
	parser.add_argument('-v', '--verbose', action="store_true")
	parser.add_argument('--ids', action="store", help=argparse.SUPPRESS)
	args = parser.parse_args()

	'''
	(opts, args) = optp.parse_args()
	if opts.verbose:
		if opts.verbose == 1:
			log_level = logging.INFO
		elif opts.verbose >= 2:
			log_level = logging.DEBUG
	'''
	log_level = logging.WARNING
	logging.getLogger().setLevel(log_level)
	# ## Setup

	settings_file = args.infile + '.learn'
	training_file = args.infile + '.train'

	args.idmapper = IDmapper()
	
	print('importing data ...')
	data_d = readData(args)

	if os.path.exists(settings_file):
		# If a settings file already exists, we'll just load that and skip training
		print('reading from', settings_file)
		with open(settings_file, 'rb') as f:
			deduper = dedupe.StaticDedupe(f)
	else:
		# ## Training
		fields = [
			{'field': 'function', 'type': 'String'}
			]

		deduper = dedupe.Dedupe(fields)
		if os.path.exists(training_file):
			print('reading labeled examples from ', training_file)
			with open(training_file, 'rb') as f:
				deduper.prepare_training(data_d, f)
		else:
			deduper.prepare_training(data_d)

		# ## Active learning
		print('starting active labeling...')
		dedupe.console_label(deduper)
		deduper.train()

		# When finished, save our training to disk
		with open(training_file, 'w') as tf:
			deduper.write_training(tf)

		# Save our weights and predicates to disk
		with open(settings_file, 'wb') as sf:
			deduper.write_settings(sf)

	# ## Clustering

	# `partition` will return sets of records that dedupe
	# believes are all referring to the same entity.


	print('clustering...')
	clustered_dupes = deduper.partition(data_d, 0.5)

	#print(clustered_dupes)
	#print('# duplicate sets', len(clustered_dupes))

	# ## Writing Results

	# Write our original data back out to a CSV with a new column called
	# 'Cluster ID' which indicates which records refer to each other.



	cluster_membership = {}
	for cluster_id, (records, scores) in enumerate(clustered_dupes):
		cluster_d = [data_d[c] for c in records]
		canonical_rep = dedupe.canonicalize(cluster_d)
		for record_id, score in zip(records, scores):
			#cluster_membership[record_id] = (cluster_id, score)
			cluster_membership[record_id] = {
            "cluster id" : cluster_id,
            "canonical representation" : canonical_rep,
            "confidence": score
			}

	for key,value in data_d.items():
		args.outfile.write(value['id'])
		args.outfile.write('\t')
		args.outfile.write(cluster_membership[key]['canonical representation']['function'])
		args.outfile.write('\n')

	'''
	# use data_d here
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
	'''




