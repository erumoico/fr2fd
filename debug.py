# -*- coding: utf-8 -*-

# ***************************************************************
# * Soubor:  debug.py                                           *
# * Datum:   2018-04-16                                         *
# * Autor:   Jan Doležal, xdolez52@stud.fit.vutbr.cz            *
# * Projekt: == Failure Rate λ(t) to Failure Density f(t) ==    *
# *          EVO - 00 - Evoluční hledání funkcí se specifickými *
# *          vlastnostmi (konzultace: J. Strnadel, L332)        *
# ***************************************************************

import sys

# Pro debugování:
import inspect
import traceback
import time
DEBUG_EN = True
#

# FUNKCE A TŘÍDY:

def printDbgEn(*args, **kwargs):
	kwargs.setdefault("file", sys.stderr)
	kwargs.setdefault("stack_num", 1)
	stack_num = kwargs.pop("stack_num")
	
	callerframerecord = inspect.stack()[stack_num]
	frame = callerframerecord[0]
	info = inspect.getframeinfo(frame)
	
	head = "(%s, %s, %s, time=%r, cpuTime=%r)" % (info.filename, info.function, info.lineno, time.time(), time.clock())
	
	print(head, ":", sep="", file=kwargs["file"])
	print("'''", file=kwargs["file"])
	print(*args, **kwargs)
	print("'''", file=kwargs["file"], flush=True)
#

def printDbg(*args, **kwargs):
	if not DEBUG_EN:
		return
	if "stack_num" not in kwargs:
		kwargs["stack_num"] = 2
	printDbgEn(*args, **kwargs)
#

def curInspect():
	callerframerecord = inspect.stack()[1]
	frame = callerframerecord[0]
	info = inspect.getframeinfo(frame)
	
	head = "(%s, %s, %s, time=%r, cpuTime=%r)" % (info.filename, info.function, info.lineno, time.time(), time.clock())
	
	return head
#

def callerInspect():
	callerframerecord = inspect.stack()[2]
	frame = callerframerecord[0]
	info = inspect.getframeinfo(frame)
	
	head = "(%s, %s, %s, time=%r, cpuTime=%r)" % (info.filename, info.function, info.lineno, time.time(), time.clock())
	
	return head
#

def curTraceback():
	formated_traceback = traceback.format_stack()[:-1]
	return "".join(formated_traceback)
#

# konec souboru debug.py
