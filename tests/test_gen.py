#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ***************************************************************
# * Soubor:  test_gen.py                                        *
# * Datum:   2018-04-16                                         *
# * Autor:   Jan Doležal, xdolez52@stud.fit.vutbr.cz            *
# * Projekt: == Failure Rate λ(t) to Failure Density f(t) ==    *
# *          EVO - 00 - Evoluční hledání funkcí se specifickými *
# *          vlastnostmi (konzultace: J. Strnadel, L332)        *
# ***************************************************************

import sys
from math import exp
import contextlib

NVAR = 1
NRAND = 100
MINRAND = -5
MAXRAND = 5
NFITCASES = 60

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

def generateTest(f, nfitcases=NFITCASES, filename="-", start=0, step=0.1):
	"""
	Uloží ve formátu pro TinyGp
	"""
	
	with smartOpen(filename, "w") as fp:
		print(NVAR, NRAND, MINRAND, MAXRAND, nfitcases+10, file=fp)
		
		x = start
		for n in range(NFITCASES):
			if n == 0:
				for m in range(10):
					print("", x, round(f(x), 10), file=fp)
					x = round(x + step/10, 3)
			print("", x, round(f(x), 10), file=fp)
			x = round(x + step, 3)

# ===== Konstantní fce =====
generateTest(lambda x: 0.1, filename="const_X~Exp(0.1)_data.txt")
generateTest(lambda x: 0.5, filename="const_X~Exp(0.5)_data.txt")
generateTest(lambda x: 1.0, filename="const_X~Exp(1.0)_data.txt")
generateTest(lambda x: 2.0, filename="const_X~Exp(2.0)_data.txt")
generateTest(lambda x: 10.0, filename="const_X~Exp(10.0)_data.txt")

# ===== Lineární fce - dobré =====
generateTest(lambda x: 0.1*x, filename="lin_0.1*x_data.txt")
generateTest(lambda x: 0.2*x+0.1, filename="lin_0.2*x+0.1_data.txt")
generateTest(lambda x: 0.5*x+1, filename="lin_0.5*x+1_data.txt")
generateTest(lambda x: x, filename="lin_x_data.txt")

# ===== Lineární fce - špatné =====
generateTest(lambda x: -0.5*x+1, filename="lin_bad_-0.5*x+1_data.txt")
generateTest(lambda x: 0.1*x-1, filename="lin_bad_0.1*x-1_data.txt")
generateTest(lambda x: 0.5*x-1, filename="lin_bad_0.5*x-1_data.txt")

# ===== Kvadratické fce =====
generateTest(lambda x: x**2, filename="kvadr_x**2_data.txt")

# ===== Kubické fce =====
generateTest(lambda x: x**3, filename="kub_x**3_data.txt")

# ===== Exponenciálně rostoucí fce =====
generateTest(lambda x: 2**(x), filename="exp_grow_2**x_data.txt")
generateTest(lambda x: exp(x), filename="exp_grow_e**x_data.txt")
generateTest(lambda x: exp(x)*0.1, filename="exp_grow_(e**x)*0.1_data.txt")
generateTest(lambda x: exp(x*0.5)-1, filename="exp_grow_e**(x*0.5)-1_data.txt")
generateTest(lambda x: exp(x*0.2)-1, filename="exp_grow_e**(x*0.2)-1_data.txt")

# ===== Exponenciálně klesající fce =====
generateTest(lambda x: 2**(-x), filename="exp_drop_2**-x_data.txt")
generateTest(lambda x: exp(-x), filename="exp_drop_e**-x_data.txt")
generateTest(lambda x: exp(-x)*0.5, filename="exp_drop_(e**-x)*0.5_data.txt")
generateTest(lambda x: exp(-x*0.5), filename="exp_drop_e**(-x*0.5)_data.txt")
generateTest(lambda x: exp(-x*0.2), filename="exp_drop_e**(-x*0.2)_data.txt")

# ===== Exponenciálně zběsilé fce =====
generateTest(lambda x: x**(x), filename="exp_x**x_data.txt")
generateTest(lambda x: x**(-x), filename="exp_x**-x_data.txt")
generateTest(lambda x: exp(x)*x, filename="exp_(e**x)*x_data.txt")
generateTest(lambda x: exp(-x)*x, filename="exp_(e**-x)*x_data.txt")
generateTest(lambda x: exp(x*x), filename="exp_e**(x*x)_data.txt")
generateTest(lambda x: exp(-x*x), filename="gauss_e**(-x*x)_data.txt") # Gaussovka
generateTest(lambda x: exp(-(x-3)*(x-3)), filename="gauss_e**(-(x-3)*(x-3))_data.txt") # posunutá Gaussovka s maximem v x=3

# ===== Fce tvaru vanové křivky =====
# http://www.sys-ev.com/reliability01.htm
bathtub1 = lambda t, k, beta, a1: k * beta * t**(beta-1) * exp(a1*t) # λ(t) = k*β*t^(β-1) * exp(a1*t), kde 0 < a1 && 0 < β < 1
bathtub2 = lambda t, k, beta, a1: k * t**(beta) * a1 * exp(a1*t) # λ(t) = k*t^β*a1 * exp(a1*t), kde 0 < a1 && -1 < β < 0
generateTest(lambda t: bathtub1(t, 1, 0.1, 0.9), start=0.01, filename="bathtub1(t,k=1,b=0.1,a=0.9)_data.txt")
generateTest(lambda t: bathtub2(t, 0.5, -0.9, 0.7), start=0.01, filename="bathtub2(t,k=0.5,b=-0.9,a=0.7)_data.txt")
generateTest(lambda t: bathtub1(t, 0.5, 0.1, 0.9), start=0.01, filename="bathtub1(t,k=0.5,b=0.1,a=0.9)_data.txt")
generateTest(lambda x: exp(x)/(x+0.01), filename="bathtub_div(exp(x),(x+0.01))_data.txt")

# konec souboru fr2fd.py
