LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: main.pdf supplementary.pdf

%.pdf: %.tex body.tex appendix.tex common.tex refs.bib $(wildcard analysis/*.tex)
	$(LATEX) $*
	bibtex $*
	$(LATEX) $*
	$(LATEX) $*

clean:
	$(RM) *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk *.synctex.gz
