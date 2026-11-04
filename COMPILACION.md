# Compilación del artículo

El archivo `IA2_main.tex` usa `\\bibliographystyle{IEEEtran}` y `\\bibliography{references}`. Compilar desde este directorio con una distribución TeX que incluya BibTeX:

```bash
pdflatex IA2_main.tex
bibtex IA2_main
pdflatex IA2_main.tex
pdflatex IA2_main.tex
```

En NixOS puede abrirse un entorno temporal con la distribución TeX completa:

```bash
nix shell nixpkgs#texliveFull -c latexmk -pdf -interaction=nonstopmode -halt-on-error IA2_main.tex
```

En el entorno de preparación de esta revisión, `pdflatex` estuvo disponible y el documento pasó dos compilaciones sintácticas sin errores, con 17 páginas. Sin embargo, el ejecutable `bibtex` no estaba disponible; por ello, las citas aparecen sin resolver en esa compilación preliminar y no se incluye como PDF final. Después de compilar con BibTeX, revisar que no queden citas `[?]`, referencias sin resolver ni entradas bibliográficas sin citar.
