import subprocess

# Define the array of languages
# ALL
# languages = ["ada","agda","alloy","antlr","applescript","assembly","augeas","awk","batchfile","bluespec","c","c-sharp","clojure","cmake","coffeescript","common-lisp","cpp","css","cuda","dart","dockerfile","elixir","elm","emacs-lisp","erlang","f-sharp","fortran","glsl","go","groovy","haskell","html","idris","isabelle","java","java-server-pages","javascript","json","julia","kotlin","lean","literate-agda","literate-coffeescript","literate-haskell","lua","makefile","maple","markdown","mathematica","matlab","ocaml","pascal","perl","php","powershell","prolog","protocol-buffer","python","r","racket","rmarkdown","ruby","rust","sas","scala","scheme","shell","smalltalk","solidity","sparql","sql","stan","standard-ml","stata","systemverilog","tcl","tcsh","tex","thrift","typescript","verilog","vhdl","visual-basic","xslt","yacc","yaml","zig"]
# CLASSIFICATION
languages = [
    "c",
    "c-sharp",
    "cpp",
    "css",
    "go",
    "html",
    "java",
    "javascript",
    "json",
    "kotlin",
    "markdown",
    "php",
    "python",
    "ruby",
    "rust",
    "scala",
    "shell",
    "sql",
    "tex",
    "typescript",
    "yaml",
]
# SPECIFIC
# languages = ["assembly"]
# Loop through each language and execute the shell script
for language in languages:
    print(f"Processing language: {language}")
    # Call the shell script with the language as an argument
    subprocess.Popen(
        [
            "sbatch",
            "./repos/data-pipeline-pretrain/examples/code_pipeline/code_pipeline.sh",
            language,
        ]
    )