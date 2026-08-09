MAIN = main

.PHONY: all draft clean

all:
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.toc *.blg *.bbl *.bcf *.out *.run.xml *.synctex.gz $(MAIN).pdf
