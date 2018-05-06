# -*- coding: utf-8 -*-
#
# Makefile pro projekt do předmětu EVO a BIN
# 
# @breif Projekt: 00 - Evoluční hledání funkcí se specifickými vlastnostmi (konzultace: J. Strnadel, L332).
# @author Jan Doležal <xdolez52@stud.fit.vutbr.cz>
#

# = Jméno přeloženého programu =
# Failure Rate λ(t) to Failure Density f(t)
login=xdolez52
program=fr2fd
prezentace=xdolez52-prezentace-00-Evolucni-hledani-funkci-se-specifickymi-vlastnostmi.pdf
pack_name=$(login)

# = Seznam zdrojových souborů =
SRC=$(program).py __init__.py my_exceptions.py debug.py
OTHER=Makefile $(prezentace) tests

# = Seznam potřebných Python balíčků =
PY_PACKAGES=sympy[alldeps] gplearn[alldeps] tkinter[alldeps]

# = Nastavení cílů bez souboru =
.PHONY: clean cleanPrezentace cleanBackup
.PHONY: pack packZip packTar
.PHONY: build all prezentace install test

# = Nastavení výchozího cíle =
.PHONY: default
default: build

# = Obecné cíle =
all: build prezentace

# = Obecné cíle pro sestavení =
build: $(program)

prezentace:
	make -C ../prezentace
	cp ../prezentace/$(prezentace) ./

# = Instalace potřebných balíků =
install:
	pip3 install --user --upgrade $(PY_PACKAGES)

# = Smazání všech dočasných souborů =
clean:
	rm -rf __pycache__
	rm -f *.pyc $(program)

cleanPrezentace:
	rm -f $(prezentace)
	make -C ../prezentace clean

cleanBackup:
	rm -f *~ *.orig

# = Zabalí program a dokumentaci =
pack: packZip

packZip: all
	rm -f $(pack_name).zip
	zip -r $(pack_name).zip $(SRC) $(HEAD) $(OTHER)

packTar: all
	rm -f $(pack_name).tar.gz
	tar -cvzf $(pack_name).tar.gz $(SRC) $(HEAD) $(OTHER)

# = Konkrétní cíle pro sestavení =
# Přidělení práv ke spuštění a vytvoření relativního symbolického odkazu
$(program):
	chmod +x $(program).py
	ln -s -r $(program).py $(program)

# = Cíle pro testování =
test: build
	true

# konec souboru Makefile
