MAIN = main

all:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	-biber $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.toc *.blg *.bbl *.bcf *.out *.run.xml *.synctex.gz $(MAIN).pdf
