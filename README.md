# Cognlp-term-paper

## Steps to reproduce the results in the term paper

(Requires a working environment for Python and R)

### 1. Clone this repository 

run `git clone https://github.com/Vincenzofan/Cognlp-term-paper.git` or click on `Code` on the top right corner and then `Dowdload ZIP`.

### 2. Install packages 

Install any packages in `requirements.txt` that are not avaliable in your environment.

The R scripts additionally require the following packages:

`tidyverse`
`emmeans`
`osfr`

Spacy additionally requires an English model, you could run `python -m spacy download en_core_web_sm` to install.

### 3. Run `main.sh` 

Change your working directory to this directory, and run `sh main.sh`.
