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
import contextlib

import sympy
import gplearn.functions

import numpy as np
#import matplotlib
#matplotlib.use("Agg") # Odkomentovat pokud nemáme Xka a/nebo nechceme zobrazovat graf.
import matplotlib.pyplot as plt
#import matplotlib.animation as animation

# Pro debugování:
import debug
from debug import printDbg
debug.DEBUG_EN = False
#

import my_exceptions

# ====== GLOBÁLNÍ PROMĚNNÉ ======

VERBOSITY = 0
def setVerbosity(verbosity):
	global VERBOSITY
	if isinstance(verbosity, int):
		VERBOSITY = verbosity
	else:
		raise TypeError("NaN")
	return VERBOSITY

# ====== KONSTANTY ======

# Získání absolutní cesty k adresáři ve kterém je tento soubor.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REGEX_NEW_LINE = re.compile(r"""(?:\r\n|[\r\n])""")

DEFAULT_FILE_SAVE_POINTS = "data.txt"
DEFAULT_FILE_FIGURE_SUFFIX = ".pdf"
DEFAULT_FILE_FIGURE_ANIM_SUFFIX = ".png"
DEFAULT_FILE_FIGURE_MOVIE_SUFFIX = ".mp4"
DEFAULT_FILE_TREE_GRAPH_SUFFIX = "-graph.graphviz"

# ====== FUNKCE A TŘÍDY ======

replaceNewLine = lambda input_string: REGEX_NEW_LINE.sub(r"""\\n """, input_string)

Coord = collections.namedtuple("Coord", "x y")

import tkinter

class CoordsGetter:
	
	def __init__(self, verbose=True):
		self.verbose = bool(verbose)
		
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
		# TODO: <Escape>: clean, <S>: save
		# TODO: V pravém horním rohu vykresvat aktuální platné souřadnice kurzoru myši.
		self.canvas.bind("<Button-1>", self.pushCoord)
		self.window.bind('<BackSpace>', self.popCoord)
		self.window.bind('<Return>', self.saveAndClose)
		self.window.protocol('WM_DELETE_WINDOW', self.saveAndClose)
		
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
			if self.verbose:
				print("Pushed", coord)
			
			# Vykreslení křížku
			self.drawCross(canvas_coord, tag=coord)
	
	def popCoord(self, event=None):
		if self.coords:
			coord = self.coords.pop()
			if self.verbose:
				print("Popped", coord)
			
			self.canvas.delete(coord)
		else:
			if self.verbose:
				print("Nothing to pop.")
	
	def save(self, event=None):
		"""
		Uloží ve formátu pro TinyGp
		"""
		filename = DEFAULT_FILE_SAVE_POINTS
		
		NVAR = 1
		NRAND = 100
		MINRAND = -5
		MAXRAND = 5
		
		with smartOpen(filename, "w") as fp:
			print(NVAR, NRAND, MINRAND, MAXRAND, len(self.coords), file=fp)
			for coord in self.coords:
				print("", coord.x, coord.y, file=fp)
	
	def close(self, event=None):
		self.window.destroy()
	
	def saveAndClose(self, event=None):
		self.save(event)
		self.close(event)
	
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
		self.canvas.create_line(-self.padding, 0, self.width, 0, fill='gray')
		
		# Vykreslení osy y
		self.canvas.create_line(0, self.padding, 0, -self.height, fill='gray')
		
		# Vykreslení hodnoty nula
		self.canvas.create_text(-4, 4, text=str(0), anchor=tkinter.NE)
		
		# Definice pomocné funkce pro tisk osy
		def _drawAxis(canvas, sign, scale, length, draw_x, width, height):
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
						canvas.create_line(x, 0, x, -height, dash=(4, 4), fill='gray') # Pro vykreslení mřížky
					else:
						x, y = y, x
						canvas.create_line(-x, y, x, y, fill='gray', capstyle = tkinter.PROJECTING)
						canvas.create_text(-x, y, text=str(y*sign // scale), anchor=tkinter.E)
						canvas.create_line(0, y, width, y, dash=(4, 4), fill='gray') # Pro vykreslení mřížky
				else:
					x *= sign
					y = comma_minor
					if draw_x:
						canvas.create_line(x, y, x, -y, fill='gray', capstyle = tkinter.PROJECTING)
						canvas.create_line(x, 0, x, -height, dash=(1, 3), fill='gray') # Pro vykreslení mřížky # TODO: Tečky pouze tam, kde se střetávají comma_minor X a Y.
					else:
						x, y = y, x
						canvas.create_line(-x, y, x, y, fill='gray', capstyle = tkinter.PROJECTING)
						canvas.create_line(0, y, width, y, dash=(1, 3), fill='gray') # Pro vykreslení mřížky
		
		# Vykreslení hodnot na ose x
		width_scale = abs(self.width_scale)
		width_sign = self.width_scale // width_scale
		_drawAxis(self.canvas, width_sign, width_scale, self.width, True, self.width, self.height)
		
		# Vykreslení hodnot na ose y
		height_scale = abs(self.height_scale)
		height_sign = self.height_scale // height_scale
		_drawAxis(self.canvas, height_sign, height_scale, self.height, False, self.width, self.height)

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

@contextlib.contextmanager
def smartOpen(filename, mode="r"):
	if filename == "-":
		if "r" in mode:
			fp = sys.stdin
		else:
			fp = sys.stdout
	else:
		fp = open(filename, mode)
	
	try:
		yield fp
	finally:
		if filename != "-":
			fp.close()

def loadCoords(filename):
	coords = []
	with smartOpen(filename) as fp:
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

def extractExprFromGplearn(program):
	"""
	Převzato z https://github.com/trevorstephens/gplearn/blob/master/gplearn/_program.py
	"""
	terminals = [0]
	output = ''
	for i, node in enumerate(program):
		if isinstance(node, gplearn.functions._Function):
			terminals.append(node.arity)
			output += node.name + '('
		else:
			if isinstance(node, int):
				output += 'X%s' % node
			else:
				# Převzato kvůli tomuto bloku.
#				output += '%.3f' % node # Tento řádek je původní.
				output += repr(node)
#				output += sympy.srepr(node)
#				output += sympy.srepr(sympy.nsimplify(node))
#				output += sympy.srepr(sympy.nsimplify(node, rational_conversion='exact', rational=True, tolerance=1e-7))
			terminals[-1] -= 1
			while terminals[-1] == 0:
				terminals.pop()
				terminals[-1] -= 1
				output += ')'
			if i != len(program) - 1:
				output += ', '
	return output

def regressionFr(coords, seed=None, population_size=None, generations=None):
	if population_size is None:
		population_size = 1000
	if generations is None:
		generations = 20
	
	# Rozdělení x-ových a y-ových souřadnic pro GpLearn
	X_train, y_train = zip(*(([x], y) for (x, y) in coords))
	
	from gplearn.genetic import SymbolicRegressor
	# Kolik náhodných čísel gplearn vygeneruje? Není omezeno. Buď se dosadí funkce, proměnná nebo se vygeneruje náhodné číslo z daného intervalu.
	est_gp = SymbolicRegressor( # Estimator
		population_size=population_size, generations=generations, tournament_size=20, stopping_criteria=0.0,
		const_range=(-5.0, 5.0), init_depth=(2, 6), init_method='half and half',
		function_set=('add', 'sub', 'mul', 'div'), metric='mean absolute error',
		parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01,
		p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05, max_samples=1.0,
		warm_start=False, n_jobs=-1, verbose=VERBOSITY, random_state=seed
	)
	est_gp.fit(X_train, y_train)
	best_individual = est_gp._program
	return est_gp, extractExprFromGplearn(best_individual.program)

def fr2fd(expression):
	from sympy import symbols, Add, Mul, Lambda, exp, Integral, sympify, lambdify
	
	a, b = symbols('a b')
	x = symbols('x', negative=False, real=True)
	t = symbols("t", negative=False, real=True)
	locals = {
		"add": Add,
		"mul": Mul,
		"sub": Lambda((a, b), a - b),
		"div": Lambda((a, b), a / b),
		"X0": x
	}
	fr = sympify(expression, locals=locals, evaluate=False) # h(x) nebo-li λ(x) nebo-li "Failure Rate"
	printDbg("h(x) =", fr)
	int_fr = Integral(fr, (x, 0, t))
	expr_int_fr = int_fr.doit()
	
	fr = fr.subs(x, t) # h(t) nebo-li λ(t) nebo-li "Failure Rate"
	printDbg("h(t) =", fr)
	rf = exp(-expr_int_fr) # R(t) = exp(-integrate(h(x),x,0,t)) nebo-li "Reliability Function"
	printDbg("R(t) =", rf)
	fd = fr * rf # f(t) = h(t)*R(t) = h(t)*exp(-integrate(h(x),x,0,t)) nebo-li "Failure Density"
	printDbg("f(t) =", fd)
	uf = 1 - rf # F(t) = 1-R(t) = 1-exp(-integrate(h(x),x,0,t)) nebo-li "Unreliability Function"
	printDbg("F(t) =", uf)
	
	printDbg(fd == uf.diff(t))
	printDbg(fr == fd / rf)
	
	# Vytvoření rychle vyhodnotitelných funkcí
	fr_lambda = lambdify(t, fr, [{"gplearnDiv": gplearn.functions.div2}, "numpy"]) # "Failure Rate"
	fd_lambda = lambdify(t, fd, [{"gplearnDiv": gplearn.functions.div2}, "numpy"]) # "Failure Density"
	uf_lambda = lambdify(t, uf, [{"gplearnDiv": gplearn.functions.div2}, "numpy"]) # "Unreliability Function"
	rf_lambda = lambdify(t, rf, [{"gplearnDiv": gplearn.functions.div2}, "numpy"]) # "Reliability Function"
	
	return {"h": fr, "f": fd, "F": uf, "R": rf, "h(t)": fr_lambda, "f(t)": fd_lambda, "F(t)": uf_lambda, "R(t)": rf_lambda}

# ====== MAIN ======

def argParse():
	"""
	Zpracování parametrů příkazového řádku
	"""
	
	parser = argparse.ArgumentParser(
		description = "A foo that bars",
		epilog = "And that's how you'd foo a bar"
	)
	
	parser.add_argument("--debug",
		action = "store_true",
		dest = "debug",
		help = ("Povolí ladící výpisy.")
	)
	parser.add_argument("-v", "--verbose",
		action = "count",
		dest = "verbosity",
		default = 0,
		help = ("Povolí výpis průběhu evoluce.")
	)
	parser.add_argument("-V", "--version",
		action = "version",
		version = "%(prog)s 0.9.0",
		help = ("Pouze vypíše verzi tohoto programu a skončí.")
	)
	
	parser.add_argument("--not_show_figure",
		action = "store_false",
		dest = "show_figure",
		help = ("Zakáže zobrazení grafu průběhu funkcí λ(t), f(t) a F(t).")
	)
	parser.add_argument("--save_figure",
		nargs = "?",
		default = False,
		const = True,
		type = str,
		help = ("Cesta s názvem souboru, do kterého se má uložit obrázek výsledného grafu. "
			"Výchozí cesta je dle vstupního souboru. Výchozí přípona je \"%s\". Existující soubor bude přepsán." % DEFAULT_FILE_FIGURE_SUFFIX
		)
	)
	parser.add_argument("--save_figure_animation",
		nargs = "?",
		default = False,
		const = True,
		type = str,
		help = ("Cesta s názvem souboru, do kterého se má uložit animace evoluce funkcí. (neimplementováno) "
			"Výchozí cesta je dle vstupního souboru. Výchozí přípona je \"%s\". Existující soubor bude přepsán." % DEFAULT_FILE_FIGURE_ANIM_SUFFIX
		)
	)
	parser.add_argument("--save_figure_movie",
		nargs = "?",
		default = False,
		const = True,
		type = str,
		help=("Cesta s názvem souboru, do kterého se má uložit video evoluce funkcí. (neimplementováno) "
			"Výchozí cesta je dle vstupního souboru. Výchozí přípona je \"%s\". Existující soubor bude přepsán." % DEFAULT_FILE_FIGURE_MOVIE_SUFFIX
		)
	)
	parser.add_argument("--save_tree_graph",
		nargs = "?",
		default = False,
		const = True,
		type = str,
		help = ("Cesta s názvem souboru, do kterého se má uložit kód pro vykreslení výsledného grafu nalezeného stromu. "
			"Výchozí cesta je dle vstupního souboru. Výchozí přípona je \"%s\". Existující soubor bude přepsán. "
			"Kód je posléze možné vykreslit, např. do PDF, pomocí: \ndot 'input.graphviz' -Tpdf > 'output.pdf'" % DEFAULT_FILE_TREE_GRAPH_SUFFIX
		)
	)
	
	parser.add_argument("-S", "--seed",
		action = "store",
		default = generateSeed(),
		type = int,
		help = ("Počáteční seed.")
	)
	
	parser.add_argument("--population_size",
		action = "store",
		default = None,
		type = int,
		help = ("Velikost populace.")
	)
	parser.add_argument("--generations",
		action = "store",
		default = None,
		type = int,
		help = ("Maximální počet generací.")
	)
	
	parser.add_argument("--gp_engine",
		action = "store",
		default = "gplearn",
		choices = ("gplearn", "tinygp", "cgp"),
		help = ("Výběr implementace genetického programování (neimplementováno – pouze \"gplearn\").")
	)
	
	parser.add_argument("file_with_points",
		nargs = "?",
		default = None,
		type = str,
		help = ("Cesta k souboru s body, jenž mají být interpolovány. "
			"Je-li zadáno \"-\", čte se ze stdin. "
			"Není-li zadán soubor s body, pak se souřadnice získají od uživatele naklikáním do grafu. "
			"Levé tlačítko myši <LMB> přidá bod, <BackSpace> odebere poslední přidaný, <Enter> uloží a ukončí. "
			"Body jsou uloženy do souboru \"%s\". Existuje-li, přepíše se." % DEFAULT_FILE_SAVE_POINTS
		),
		metavar = DEFAULT_FILE_SAVE_POINTS,
	)
	arguments = parser.parse_args()
	
	return arguments

def main():
	
	# == Abychom věděli co šahá kam nemá ==
	signal.signal(signal.SIGSEGV, my_exceptions.signalHandler)
	
	# == Signály, které ukončují program ==
	#signal.signal(signal.SIGQUIT, my_exceptions.signalHandler)
	#signal.signal(signal.SIGTERM, my_exceptions.signalHandler)
	signal.signal(signal.SIGINT, my_exceptions.signalHandler)
	
	# == Zpracování parametrů příkazového řádku ==
	arguments = argParse()
	
	# == Povolení ladících výpisů ==
	debug.DEBUG_EN = arguments.debug
	
	# == Nastavení ukecaného výstupu ==
	setVerbosity(arguments.verbosity)
	
	# == Načtení souřadnic pro symbolickou regresi ==
	# Není-li zadán soubor s body, pak se souřadnice získají pomocí třídy CoordsGetter
	if arguments.file_with_points is not None:
		coords = loadCoords(arguments.file_with_points)
	else:
		coords_getter = CoordsGetter(verbose = VERBOSITY)
		coords = coords_getter.getCoords()
	printDbg(coords)
	
	# == Symbolická regrese za pomocí genetického programování ==
	seed = arguments.seed % 2**32 # Takto to vyžaduje gplearn.
	print("seed =", seed)
	fr, fr_str = regressionFr(coords,
		seed = seed,
		population_size = arguments.population_size,
		generations = arguments.generations)
	print("h(X0) =", fr_str)
	
	# == uložení kódu pro vykreslení výsledného grafu nalezeného stromu do souboru ==
	if arguments.file_with_points is not None:
		default_filename_prefix = os.path.splitext(arguments.file_with_points)[0]
	else:
		default_filename_prefix = os.path.splitext(DEFAULT_FILE_SAVE_POINTS)[0]
	
	if arguments.save_tree_graph:
		filename = arguments.save_tree_graph
		filename = filename is not True and filename or default_filename_prefix + DEFAULT_FILE_TREE_GRAPH_SUFFIX
		with smartOpen(filename, "w") as fp:
			print(fr._program.export_graphviz(), file=fp) # tisk kódu do souboru
	
	# == Integrace fce h(t) za pomocí symbolického výpočtu ==
	results = fr2fd(fr_str)
	
	# == Zobrazení výsledných funkcí ==
	for f in ("h", "f", "F"):
		print(f+"(t)", "=", results[f])
	
	# == Zobrazení grafů výsledných funkcí pomocí matplotlib.pyplot as plt ==
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
	fig.subplots_adjust(left=0.2, wspace=0.6)
	
	# vykreslení vstupních bodů
	input_x, input_y = zip(*coords)
	ax1.plot(input_x, input_y, color='black', marker='x', linestyle="")
	
	# = příprava x-ových souřadnic pro vykreslení grafů =
	min_x = coords[0].x # minimální x-ová souřadnice
	max_x = coords[-1].x # maximální x-ová souřadnice
	
	# regrese 10% na každou stranu
	range_min_x = min_x - (max_x - min_x)*0.1
	if range_min_x < 0:
		range_min_x = 0
	range_max_x = max_x + (max_x - min_x)*0.1
	
	# vytvoření x-ových souřadnic
	x = np.arange(range_min_x, range_max_x, 0.01)
	
	# = vykreslení fce h(t) =
	# vytvoření y-ových souřadnic pomocí gplearn.predict h(t)
	y = fr.predict(np.c_[x])
	ax1.plot(x, y, linewidth=2)
	
	# vytvoření y-ových souřadnic pomocí sympy.lambdify h(t)
	y = results["h(t)"](x)
	if isinstance(y, float): # Je-li h(t) = konst., pak y je float a ne np.array. Takže se musí namnožit.
		y = np.repeat(y, len(x))
	ax1.plot(x, y, linewidth=2, linestyle="--")
	
	font = {'family': 'sans-serif'}
	ylabel = r'$\mathsf{\lambda}(t)$'
#	title = r'Failure Rate'
	title = r'Intenzita poruch'
	ax1.set_title(" ".join((title, ylabel)), fontdict=font)
	ax1.set_ylabel(ylabel, fontdict=font)
	ax1.grid(True)
	
	# = vykreslení fce f(t) =
	# vytvoření y-ových souřadnic pomocí sympy.lambdify f(t)
	y = results["f(t)"](x)
	ax2.plot(x, y, linewidth=2)
	ylabel = r'$\mathsf{f}(t)$'
#	title = r'Failure Density'
	title = r'Hustota pravděpodobnosti poruchy'
	ax2.set_title(" ".join((title, ylabel)), fontdict=font)
	ax2.set_ylabel(ylabel, fontdict=font)
	ax2.grid(True)
	
	# = vykreslení fce F(t) =
	# vytvoření y-ových souřadnic pomocí sympy.lambdify F(t)
	y = results["F(t)"](x)
	ax3.plot(x, y, linewidth=2)
	ylabel = r'$\mathsf{F}(t)$'
#	title = r'Unreliability Function'
	title = r'Pravděpodobnost poruchy'
	ax3.set_title(" ".join((title, ylabel)), fontdict=font)
	ax3.set_ylabel(ylabel, fontdict=font)
	ax3.set_xlabel(r'$t$', fontdict=font)
	ax3.grid(True)
	
	plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
	
	# uložení grafu
	if arguments.save_figure:
		filename = arguments.save_figure
		filename = filename is not True and filename or default_filename_prefix + DEFAULT_FILE_FIGURE_SUFFIX
		plt.savefig(filename)
	
	# zobrazení grafu
	if arguments.show_figure:
		plt.show()

if __name__ == '__main__':
	main()

# konec souboru fr2fd.py
