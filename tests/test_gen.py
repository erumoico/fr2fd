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

NVAR = 1
NRAND = 100
MINRAND = -5
MAXRAND = 5
NFITCASES = 60

print(NVAR, NRAND, MINRAND, MAXRAND, NFITCASES+10)

from math import exp

# ===== Konstantní fce =====
f = lambda x: 0.1 # X~Exp(0.1)
f = lambda x: 0.5 # X~Exp(0.5)
f = lambda x: 1.0 # X~Exp(1.0)
f = lambda x: 2.0 # X~Exp(2.0)
f = lambda x: 10.0 # X~Exp(10.0)

# ===== Lineární fce - dobré =====
f = lambda x: 0.1*x
f = lambda x: 0.2*x+0.1
f = lambda x: 0.5*x+1
f = lambda x: x

# ===== Lineární fce - špatné =====
f = lambda x: -0.5*x+1
f = lambda x: 0.1*x-1
f = lambda x: 0.5*x-1

# ===== Kvadratické fce =====
f = lambda x: x**2

# ===== Kubické fce =====
f = lambda x: x**3

# ===== Exponenciálně rostoucí fce =====
f = lambda x: 2**(x) # 2**x
f = lambda x: exp(x) # e**x
f = lambda x: exp(x)*0.1 # (e**x)*0.1
f = lambda x: exp(x*0.5)-1 # e**(x*0.5)-1
f = lambda x: exp(x*0.2)-1 # e**(x*0.2)-1

# ===== Exponenciálně klesající fce =====
f = lambda x: 2**(-x) # 2**-x
f = lambda x: exp(-x) # e**-x
f = lambda x: exp(-x)*0.5 # (e**-x)*0.5
f = lambda x: exp(-x*0.5) # e**(-x*0.5)
f = lambda x: exp(-x*0.2) # e**(-x*0.2)

# ===== Exponenciálně zběsilé fce =====
f = lambda x: x**(x) # x**x
f = lambda x: x**(-x) # x**-x
f = lambda x: exp(x)*x # (e**x)*x
f = lambda x: exp(-x)*x # (e**-x)*x
f = lambda x: exp(x*x) # e**(x*x)
f = lambda x: exp(-x*x) # e**(-x*x) -- Gaussovka
f = lambda x: exp(-(x-3)*(x-3)) # e**(-(x-3)*(x-3)) -- posunutá Gaussovka s maximem v x=3

# ===== Fce tvaru vanové křivky =====
# http://www.sys-ev.com/reliability01.htm
bathtub1 = lambda t, k, beta, a1: k * beta * t**(beta-1) * exp(a1*t) # λ(t) = k*β*t^(β-1) * exp(a1*t), kde 0 < a1 && 0 < β < 1
bathtub2 = lambda t, k, beta, a1: k * t**(beta) * a1 * exp(a1*t) # λ(t) = k*t^β*a1 * exp(a1*t), kde 0 < a1 && -1 < β < 0
f = lambda t: bathtub1(t, 1, 0.1, 0.9) # bathtub1(t,k=1,b=0.1,a=0.9)
f = lambda t: bathtub2(t, 0.5, -0.9, 0.7) # bathtub2(t,k=0.5,b=-0.9,a=0.7)
f = lambda t: bathtub1(t, 0.5, 0.1, 0.9) # bathtub1(t,k=0.5,b=0.1,a=0.9)
f = lambda x: exp(x)/(x+0.01) # bathtub_div(exp(x),(x+0.01))

x1 = 0.01
for n in range(NFITCASES):
	if n == 0:
		for m in range(10):
			print("", x1, round(f(x1), 10))
			x1 = round(x1 + 0.01, 3)
	print("", x1, round(f(x1), 10))
	x1 = round(x1 + 0.1, 3)

# konec souboru fr2fd.py
