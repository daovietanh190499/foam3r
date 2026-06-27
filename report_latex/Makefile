# Build Voronoi render report (pdflatex + babel-vietnamese)
MAIN = report
LATEX = pdflatex

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex ../data/report_figures/python/*.png ../data/report_figures/cuda/*.png
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(LATEX) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf
