# -*- coding: utf-8 -*-
#
# Makefile pro projekt do předmětu EVO a BIN
# 
# @breif Projekt: 00 - Evoluční hledání funkcí se specifickými vlastnostmi (konzultace: J. Strnadel, L332).
# @author Jan Doležal <xdolez52@stud.fit.vutbr.cz>
#

# = Jméno přeloženého programu =
login=xdolez52
program=fr2fd # Failure Rate λ(t) to Failure Density f(t)
doc_name=dokumentace
pack_name=$(login)

# Seznam zdrojových souborů
SRC=$(program).py __init__.py my_exceptions.py debug.py
OTHER=Makefile $(doc_name).pdf

# Seznam potřebných Python balíčků
PY_PACKAGES=scikit-learn[alldeps] gplearn[alldeps] inspyred[alldeps]
#inspyred terminaltables

.PHONY: build all doc install clean clean_doc clean_all pack pack_zip pack_tar test

# Zkompiluje program (výchozí)
build: $(program)

all: build doc

doc:
	make -C doc
	cp doc/$(doc_name).pdf ./

# Nainstaluje potřebné balíky
install:
	pip3 install --user --upgrade $(PY_PACKAGES)

# Smaže všechny soubory co nemají být odevzdány
clean:
	rm -f *.pyc $(program)

clean_doc:
	make -C doc clean

# Zabalí program a dokumentaci
pack: all pack_zip

pack_zip: all
	rm -f $(pack_name).zip
	zip -r $(pack_name).zip $(SRC) $(HEAD) $(OTHER)

pack_tar: all
	rm -f $(pack_name).tar.gz
	tar -cvzf $(pack_name).tar.gz $(SRC) $(HEAD) $(OTHER)

# PROGRAM
# =======

# Přidělení práv ke spuštění a vytvoření relativního symbolického odkazu
$(program):
	chmod +x $(program).py
	ln -s -r $(program).py $(program)

# TEST
# ----
# ls -1b "${PWD}/../priklady/emaily/"*.[eE][mM][lL] | sed 's/^/ham:/' > labeled_emails.txt
# ls -1b "${PWD}/../priklady/spamy/"*.[eE][mM][lL] | sed 's/^/spam:/' >> labeled_emails.txt

# Spustí testy
# test: $(program)
# 	wget -N --no-check-certificate 'https://www.fit.vutbr.cz/study/courses/BMS/public/proj2017/bts.csv'

# konec souboru Makefile
