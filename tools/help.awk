# Print `make help` from the Makefile. A `##@ Title` line opens a section, a `target: ## text`
# line prints one line of the help, and a `## text` line joins the variables at the end.

BEGIN {
    FS = ":.*## "
    # system() shares the stdout of awk, so `test -t 1` sees the terminal of `make help`.
    if (system("test -t 1") == 0 && !("NO_COLOR" in ENVIRON)) {
        bold = "\033[1m"
        cyan = "\033[36m"
        reset = "\033[0m"
    }
    print "zigferencer make targets"
}

/^##@ / { printf "\n%s%s%s\n", bold, substr($0, 5), reset; next }
/^## / { variables = variables "  " substr($0, 4) "\n"; next }
/^[a-z][a-z-]*:.*## / { printf "  %s%-14s%s %s\n", cyan, $1, reset, $2 }

END { printf "\n%sVariables%s\n%s", bold, reset, variables }
