# -*- coding: utf-8 -*-

# ***************************************************************
# * Soubor:  my_exceptions.py                                   *
# * Datum:   2018-04-16                                         *
# * Autor:   Jan Doležal, xdolez52@stud.fit.vutbr.cz            *
# * Projekt: == Failure Rate λ(t) to Failure Density f(t) ==    *
# *          EVO - 00 - Evoluční hledání funkcí se specifickými *
# *          vlastnostmi (konzultace: J. Strnadel, L332)        *
# ***************************************************************

import signal
import inspect

# KONSTANTY:

SIGNALS_TO_NAMES_DICT = dict(
	(k, v) for v, k in sorted(signal.__dict__.items(), reverse=True) \
		if v.startswith('SIG') and not v.startswith('SIG_')
)

# FUNKCE A TŘÍDY:

class AntispamException(Exception):
	def __str__(self):
		return "%s: %s" % (self.__class__.__name__, super(AntispamException, self).__str__())
	pass
#

class SpamDetect(AntispamException):
	pass
#

class InternalError(AntispamException):
	pass
#

class TimeoutError(AntispamException):
	pass
#

class SubprocessError(AntispamException):
	pass
#

class CatchedSignal(AntispamException):
	def __init__(self, signum, frame):
		self.signum = signum
		self.frameinfo = inspect.getframeinfo(frame)
	
	def __str__(self):
		# SIGSEGV "Segmentation fault"
		return "%s: Catched signal %s from (%s, %s, %d)." % (self.__class__.__name__, getSignalName(self.signum), self.frameinfo.filename, self.frameinfo.function, self.frameinfo.lineno)
#

def getSignalName(signum):
	return SIGNALS_TO_NAMES_DICT.get(signum, "Unnamed signal: %d" % signum)

def signalHandler(signum, frame):
    raise CatchedSignal(signum, frame)

# konec souboru my_exceptions.py
