#!/usr/bin/env python3
# -*- coding: utf-8 -*-

NVAR = 1
NRAND = 100
MINRAND = -5
MAXRAND = 5
NFITCASES = 60

print(NVAR, NRAND, MINRAND, MAXRAND, NFITCASES+10)

# X~Exp(0.5)
x1 = 0
TARGET = 0.5
for n in range(NFITCASES):
	if n == 0:
		for m in range(10):
			print("", round(x1, 3), TARGET)
			x1 += 0.01
	print("", round(x1, 3), TARGET)
	x1 += 0.1

# EOF
