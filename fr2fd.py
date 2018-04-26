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
import os, sys
import signal
import re
import time, random
import collections

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

Coord = collections.namedtuple("Coord", "x y")

import tkinter

class CoordsGetter:
	
	def __init__(self):
		self.width = 600+1
		self.width_scale = 50 # počet pixelů na jednotku na ose x
		self.height = 400+1
		self.height_scale = -200 # počet pixelů na jednotku na ose y
		self.padding = 20
		
		self.coords = []
		self.window = tkinter.Tk(className = "CoordsGetter")
		self.canvas = tkinter.Canvas(self.window,
			width = self.width + self.padding,
			height = self.height + self.padding,
			borderwidth = 0,
			xscrollincrement = 1,
			yscrollincrement = 1,
		)
		self.canvas.scan_dragto(1 + self.padding, self.height, gain=1)
		self.canvas.pack()
		
		# Nastavení událostí na tlačítka
		self.canvas.bind("<Button-1>", self.pushCoord)
		self.window.bind('<BackSpace>', self.popCoord)
		self.window.bind('<Return>', self.close)
		self.window.protocol('WM_DELETE_WINDOW', self.close)
		
		# Vykreslení souřadné osy
		self.drawAxes()
	
	def getCoords(self):
		# Čekání ve smyčce
		self.window.mainloop()
		
		# Vrácení souřadnic
		return self.coords
	
	def pushCoord(self, event):
		x = self.canvas.canvasx(event.x)
		if x <= 0.0:
			x = 0.0
		y = self.canvas.canvasy(event.y)
		if y >= 0.0:
			y = 0.0
		canvas_coord = Coord(x, y)
		
		x = canvas_coord.x and canvas_coord.x / self.width_scale
		y = canvas_coord.y and canvas_coord.y / self.height_scale
		coord = Coord(x, y)
		
		# Definiční obor i obor hodnot je od nuly do nekonečna a žádné dva body nesmí sdílet x-ovou souřadnici
		if coord.x >= 0.0 and coord.y >= 0.0 and (not self.coords or coord.x > self.coords[-1].x):
			self.coords.append(coord)
			print("Pushed", coord)
			
			# Vykreslení křížku
			self.drawCross(canvas_coord, tag=coord)
	
	def popCoord(self, event=None):
		if self.coords:
			coord = self.coords.pop()
			print("Popped", coord)
			
			self.canvas.delete(coord)
		else:
			print("Nothing to pop.")
	
	def close(self, event=None):
		self.window.destroy()
	
	def drawCross(self, coord, tag=None, cross_length = 3):
		"""
		Vykreslení křížku
		"""
		first_line = self.canvas.create_line(coord.x - cross_length, coord.y - cross_length, coord.x + cross_length, coord.y + cross_length, capstyle = tkinter.PROJECTING)
		second_line = self.canvas.create_line(coord.x - cross_length, coord.y + cross_length, coord.x + cross_length, coord.y - cross_length, capstyle = tkinter.PROJECTING)
		if tag is not None:
			self.canvas.addtag_withtag(tag, first_line)
			self.canvas.addtag_withtag(tag, second_line)
	
	def drawAxes(self):
		"""
		Vykreslení souřadné osy
		"""
		# Vykreslení osy x
		self.canvas.create_line(-self.padding, 0, self.width, 0, dash=(2, 2), fill='gray')
		
		# Vykreslení osy y
		self.canvas.create_line(0, self.padding, 0, -self.height, dash=(2, 2), fill='gray')
		
		# Vykreslení hodnoty nula
		self.canvas.create_text(-4, 4, text=str(0), anchor=tkinter.NE)
		
		# Definice pomocné funkce pro tisk osy
		def _drawAxis(canvas, sign, scale, length, draw_x):
			comma_major = 4
			comma_minor = 2
			if scale < 25:
				major = 50 // scale
				for d in (5, 2, 1):
					if major % d == 0:
						minor = major // d
						break
				minor *= scale
				major *= scale
			else:
				major = 1
				if scale < 50:
					possible_minors = (5, 2, 1)
				else:
					possible_minors = (10, 5, 2, 1)
				for d in possible_minors:
					if scale % d == 0:
						minor = d
						break
				minor = scale // minor
				major = scale
			for x in range(minor, length, minor):
				if x % major == 0:
					x *= sign
					y = comma_major
					if draw_x:
						canvas.create_line(x, y, x, -y, fill='gray', capstyle = tkinter.PROJECTING)
						canvas.create_text(x, y, text=str(x*sign // scale), anchor=tkinter.N)
					else:
						x, y = y, x
						canvas.create_line(-x, y, x, y, fill='gray', capstyle = tkinter.PROJECTING)
						canvas.create_text(-x, y, text=str(y*sign // scale), anchor=tkinter.E)
				else:
					x *= sign
					y = comma_minor
					if draw_x:
						canvas.create_line(x, y, x, -y, fill='gray', capstyle = tkinter.PROJECTING)
					else:
						x, y = y, x
						canvas.create_line(-x, y, x, y, fill='gray', capstyle = tkinter.PROJECTING)
		
		# Vykreslení hodnot na ose x
		width_scale = abs(self.width_scale)
		width_sign = self.width_scale // width_scale
		_drawAxis(self.canvas, width_sign, width_scale, self.width, True)
		
		# Vykreslení hodnot na ose y
		height_scale = abs(self.height_scale)
		height_sign = self.height_scale // height_scale
		_drawAxis(self.canvas, height_sign, height_scale, self.height, False)

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

def loadCoords(filename):
	coords = []
	with open(filename, 'r') as fp:
		x_coords = set()
		for line in fp: # Načítají se body
			values = line.split()
			if len(values) == 2: # Nastavení pro TinyGp se přeskočí -- má jiný počet hodnot
				coord = Coord(*map(float, values))
				if coord.x not in x_coords: # Zabraňuje tomu, aby libovolné dva body sdíleli x-ovou souřadnici
					x_coords.add(coord.x)
					coords.append(coord)
				else:
					print("Bad coord", coord, "in file", filename, "-->", "skipped", file=sys.stderr)
	sorted_coords = sorted(coords)
	if sorted_coords != coords: # Souřadnice by měly být seřazeny dle X-ové souřadnice
		print("Coords", "in file", filename, "were not sort by X coord --> sorted", file=sys.stderr)
	
	return sorted_coords

def regressionFr(coords, seed=None, population_size=None, generations=None):
	if population_size is None:
		population_size = 1000
	if generations is None:
		generations = 20
	
	# Rozdělení x-ových a y-ových souřadnic pro GpLearn
	X_train, y_train = zip(*(([x], y) for (x, y) in coords))
	
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

# ====== MAIN ======

def main():
	
	# == Abychom věděli co šahá kam nemá ==
	signal.signal(signal.SIGSEGV, my_exceptions.signalHandler)
	
	# == Signály, které ukončují program ==
	#signal.signal(signal.SIGQUIT, my_exceptions.signalHandler)
	#signal.signal(signal.SIGTERM, my_exceptions.signalHandler)
	signal.signal(signal.SIGINT, my_exceptions.signalHandler)
	
	# == Zpracování parametrů ==
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
		nargs="?",
		default=None,
		type=str,
		help=("Cesta k souboru s body, jenž mají být interpolovány."),
		metavar="data.txt",
	)
	arguments = parser.parse_args()
	
	# == Povolení ladících výpisů ==
	debug.DEBUG_EN = arguments.debug
	
	# == Načtení souřadnic pro symbolickou regresi ==
	# Není-li zadán soubor s body, pak se souřadnice získají pomocí třídy CoordsGetter
	if arguments.file_with_points is not None:
		coords = loadCoords(arguments.file_with_points)
	else:
		coords_getter = CoordsGetter()
		coords = coords_getter.getCoords()
	print(coords)
	
	# == Symbolická regrese za pomocí genetického programování ==
	seed = arguments.seed % 2**32 # Takto to vyžaduje gplearn.
	print("seed =", seed)
	fr_str = regressionFr(coords,
		seed = seed,
		population_size = arguments.population_size,
		generations = arguments.generations)
	print("h(t) =", fr_str)
	
	# == Integrace za pomocí symbolického výpočtu ==
	results = fr2fd(fr_str)
	
	# == Zobrazení výsledných funkcí ==
	for f, expr in results.items():
		print(f, "=", expr)
	
	# == Zobrazení grafů výsledných funkcí pomocí sympy ==
	t = sympy.symbols('t')
	for f, expr in results.items():
		sympy.plot(expr, (t, coords[0].x, coords[-1].x), title=f, ylabel=f)
	
	# == Zobrazení grafů výsledných funkcí pomocí matplotlib.pyplot ==
#	ax = plt.subplot(111)
#	t = np.arange(min(X_train)-(max(X_train)-min(X_train))*0.1, max(X_train)+(max(X_train)-min(X_train))*0.1, 0.01)
#	s1 = np.exp(1*t)
#	plt.plot(t, s1, lw=2)
#	font = {'family': 'serif', 'weight': 'normal', 'size': 22}
#	plt.title(r'$y = e^{kx}$', fontdict=font)
#	plt.grid(True)
#	plt.show()

if __name__ == '__main__':
	main()

# konec souboru fr2fd.py
