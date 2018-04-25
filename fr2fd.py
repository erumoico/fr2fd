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
import collections

#from gplearn.genetic import SymbolicRegressor
import sympy

#import numpy as np
#import matplotlib.pyplot as plt

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

import tkinter

class CoordsGetter:
	Coord = collections.namedtuple("Coord", "x y")
	
	def __init__(self):
		self.width = 100
		self.width_scale = 1
		self.canvas_width = self.width * self.width_scale
		self.height = 100
		self.height_scale = 1
		self.canvas_height = self.height * self.height_scale
		
		self.coords = []
		self.window = tkinter.Tk(className = "CoordsGetter")
		self.canvas = tkinter.Canvas(self.window, width = self.canvas_width, height = self.canvas_height)
		self.canvas.pack()
		
		# Nastavení událostí na tlačítka
		self.canvas.bind("<Button-1>", self.pushCoord)
		self.window.bind('<BackSpace>', self.popCoord)
		self.window.bind('<Return>', self.close)
		self.window.protocol('WM_DELETE_WINDOW', self.close)
		
		# Vykreslení souřadné osy
		self.drawAxis()
	
	def getCoords(self):
		# Čekání ve smyčce
		self.window.mainloop()
		
		# Vrácení souřadnic
		return self.coords
	
	def pushCoord(self, event):
		event_coord = self.Coord(event.x, event.y)
		x = event_coord.x // self.width_scale
		y = event_coord.y // self.height_scale
		coord = self.Coord(x, y)
		self.coords.append(coord)
		print("Pushed", coord)
		
		# Vykreslení křížku
		self.drawCross(event_coord, tag=coord)
	
	def popCoord(self, event=None):
		if self.coords:
			coord = self.coords.pop()
			print("Popped", coord)
			
			self.canvas.delete(coord)
		else:
			print("Nothing to pop.")
	
	def close(self, event=None):
		self.window.destroy()
	
	def drawCross(self, coord, tag=None, cross_length = 2):
		"""
		Vykreslení křížku
		"""
		first_line = self.canvas.create_line(coord.x - cross_length, coord.y - cross_length, coord.x + (cross_length+1), coord.y + (cross_length+1))
		second_line = self.canvas.create_line(coord.x - cross_length, coord.y + cross_length, coord.x + (cross_length+1), coord.y - (cross_length+1))
		if tag is not None:
			self.canvas.addtag_withtag(tag, first_line)
			self.canvas.addtag_withtag(tag, second_line)
	
	def drawAxis(self):
		"""
		Vykreslení souřadné osy
		"""
		self.canvas.create_line(0, 0, 0, self.canvas_height, dash=(2,2))
		self.canvas.create_line(0, 0, self.canvas_width, 0, dash=(2,2))

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
			X_train.append(list(map(float, coords[:-1])))
			y_train.append(float(coords[-1]))
	
	return (X_train, y_train)

def regressionFr(X_train, y_train, seed=None, population_size=None, generations=None):
	if population_size is None:
		population_size = 1000
	if generations is None:
		generations = 20
	
	from gplearn.genetic import SymbolicRegressor
	est_gp = SymbolicRegressor(
		population_size=population_size, generations=generations, tournament_size=20, stopping_criteria=0.0,
		const_range=(-5.0, 5.0), init_depth=(2, 6), init_method='half and half',
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
	
	# gplearn 'div' : protected division where a denominator near-zero returns 1.
	from sympy import Function, S
	class gplearnDiv(Function):
		@classmethod
		def eval(cls, x, y):
			if y.is_Number:
				if abs(y) <= 0.001:
					return S.One
				else:
					return x/y
			elif x.is_Symbol and x is y:
				return S.One
	
	x, y = symbols('x y')
	t = symbols("t")
	locals = {
		"add": Add,
		"mul": Mul,
		"sub": Lambda((x, y), x - y),
		"div": gplearnDiv,
		"X0": t
	}
	fr = sympify(expression, locals=locals, evaluate=False) # h(t) nebo-li λ(t)
	#fr = sympy_parse_expr(expression, local_dict=locals, evaluate=False) # h(t) nebo-li λ(t)
	rf = exp(-integrate(fr, (t, 0, t))) # R(t) = exp(-integrate(h(t),t,0,t))
	fd = fr * rf # f(t) = h(t)*R(t) = h(t)*exp(-integrate(h(t),t,0,t))
	uf = 1 - rf # F(t) = 1-R(t) = 1-exp(-integrate(h(t),t,0,t))
	
	printDbg(fd == uf.diff(t))
	printDbg(fr == fd / rf)
	
	return {"h(t)": fr, "f(t)": fd, "F(t)": uf, "R(t)": rf}

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
	seed = arguments.seed % 2**32 # Takto to vyžaduje gplearn.
	print("seed =", seed)
	X_train, y_train = loadPoints(arguments.file_with_points)
	fr_str = regressionFr(X_train, y_train, seed,
		population_size = arguments.population_size,
		generations = arguments.generations)
	print("h(t) =", fr_str)
	
#	ax = plt.subplot(111)
#	t = np.arange(min(X_train)-(max(X_train)-min(X_train))*0.1, max(X_train)+(max(X_train)-min(X_train))*0.1, 0.01)
#	s1 = np.exp(1*t)
#	plt.plot(t, s1, lw=2)
#	font = {'family': 'serif', 'weight': 'normal', 'size': 22}
#	plt.title(r'$y = e^{kx}$', fontdict=font)
#	plt.grid(True)
#	plt.show()
	
	results = fr2fd(fr_str)
	for f, expr in results.items():
		print(f, "=", expr)
	sympy.plot(results["h(t)"])
	sympy.plot(results["f(t)"])
	sympy.plot(results["F(t)"])

if __name__ == '__main__':
	main()

# konec souboru fr2fd.py
