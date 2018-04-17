#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ***************************************************************
# * Soubor:  fr2fd.py                                           *
# * Datum:   2018-04-16                                         *
# * Autor:   Jan Doležal, xdolez52@stud.fit.vutbr.cz            *
# * Projekt: == Failure Rate λ(t) to Failure Density f(t) ==    *
# *          EVO - 00 - Evoluční hledání funkcí se specifickými *
# *          vlastnostmi (konzultace: J. Strnadel, L332)        *
# ***************************************************************

import argparse
import os
import signal
import re
import time, random

import gplearn
#import sympy

# Pro debugování:
import debug
from debug import printDbg
debug.DEBUG_EN = False
#

import my_exceptions

# ====== GLOBÁLNÍ PROMĚNNÉ ======

# ====== KONSTANTY ======

# Získání absolutní cesty k adresáři ve kterém je tento soubor.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REGEX_NEW_LINE = re.compile(r"""(?:\r\n|[\r\n])""")

# ====== FUNKCE A TŘÍDY ======

replaceNewLine = lambda input_string: REGEX_NEW_LINE.sub(r"""\\n """, input_string)

# ====== MAIN ======

def uniqifyList(seq, order_preserving=False):
	if order_preserving:
		seen = set()
		return [x for x in seq if x not in seen and not seen.add(x)]
	else:
		return list(set(seq))

def generateSeed():
	try:
		seed = hash(os.urandom(32))
	except NotImplementedError:
		seed = time.time() + os.getpid()
	
	return seed

def preparePrng(prng=None, seed=None):
	if seed is None:
		seed = generateSeed()
	
	if prng is None:
		prng = random.Random()
	prng.seed(seed)
	
	return prng

def loadPoints(filename):
	X_train = []
	y_train = []
	with open(filename, 'r') as fp:
		fp.readline() # Nastavení pro TinyGp se přeskočí
		for line in fp: # Načítají se body
			coords = line.split()
			X_train.append(coords[:-1])
			y_train.append(coords[-1])
	
	return (X_train, y_train)

def regressionFr(X_train, y_train, seed=None, population_size=1000, generations=20):
	est_gp = gplearn.genetic.SymbolicRegressor(
		population_size=population_size, generations=generations, tournament_size=20, stopping_criteria=0.0,
		const_range=(-1.0, 1.0), init_depth=(2, 6), init_method='half and half',
		function_set=('add', 'sub', 'mul', 'div'), metric='mean absolute error',
		parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01,
		p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05, max_samples=1.0,
		warm_start=False, n_jobs=-1, verbose=1, random_state=seed
	)
	est_gp.fit(X_train, y_train)
	return est_gp._program

def fr2fd(expression):
	from sympy import symbols, Add, Mul, Lambda, exp, integrate, sympify
	#from sympy.parsing.sympy_parser import parse_expr as sympy_parse_expr
	x, y = symbols('x y')
	t = symbols("X0")
	locals = {
		"add": Add,
		"mul": Mul,
		"sub": Lambda((x, y), x - y),
		"div": Lambda((x, y), x/y),
		"X0": t
	}
	fr = sympify(expression, locals=locals, evaluate=False) # h(t) nebo-li λ(t)
	#fr = sympy_parse_expr(expression, local_dict=locals, evaluate=False) # h(t) nebo-li λ(t)
	rf = exp(-integrate(fr, (t, 0, t))) # R(t) = exp(-integrate(h(t),t,0,t))
	fd = fr * rf # f(t) = h(t)*R(t) = h(t)*exp(-integrate(h(t),t,0,t))
	uf = 1 - rf # F(t) = 1-R(t) = 1-exp(-integrate(h(t),t,0,t))
	
	printDbg(fd == uf.diff(t))
	printDbg(fr == fd / rf)
	
	return (fr, fd, uf, rf)

def main():
	
	# Abychom věděli co šahá kam nemá
	signal.signal(signal.SIGSEGV, my_exceptions.signalHandler)
	
	# Signály, které ukončují program
	#signal.signal(signal.SIGQUIT, my_exceptions.signalHandler)
	#signal.signal(signal.SIGTERM, my_exceptions.signalHandler)
	signal.signal(signal.SIGINT, my_exceptions.signalHandler)
	
	# Zpracování parametrů
	parser = argparse.ArgumentParser()
	parser.add_argument("--debug",
		action = "store_true",
		dest = "debug",
		help = ("Povolí ladící výpisy.")
	)
	parser.add_argument("-S", "--seed",
		action = "store",
		default = generateSeed(),
		type = int,
		help = ("Počáteční seed.")
	)
	
	parser.add_argument("--population_size",
		action="store",
		default=None,
		type=int,
		help=("Velikost populace.")
	)
	parser.add_argument("--generations",
		action="store",
		default=None,
		type=int,
		help=("Maximální počet generací.")
	)
	
	parser.add_argument("file_with_points",
		type=str,
		help=("Cesta k souboru s body, jenž mají být interpolovány."),
		metavar="data.txt",
	)
	arguments = parser.parse_args()
	
	# Povolení ladících výpisů
	debug.DEBUG_EN = arguments.debug
	
	# == Symbolická regrese za pomocí genetického programování a následná integrace za pomocí symbolického výpočtu ==
	print("seed =", arguments.seed)
	X_train, y_train = loadPoints(arguments.file_with_points)
	fr_str = regressionFr(X_train, y_train, arguments.seed)
	for expr in fr2fd(fr_str):
		print(expr)

if __name__ == '__main__':
	main()

# konec souboru fr2fd.py
