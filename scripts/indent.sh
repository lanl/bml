#!/bin/bash

: ${EMACS:=$(command -v emacs)}
: ${INDENT_ARGS:="-kr -bl -nut -i4 -bli0 -cli4 -ppi0 -cbi0 -npcs -bfda -nfc1 -nfca -ncp1"}

declare -a SH_FILES
declare -a C_FILES
declare -a FORTRAN_FILES

if ! git status > /dev/null; then
    cat <<EOF
We are not inside a git repository. Please specify the files to indent
manually.
EOF
    exit 1
fi

check_only=0

if (( $# > 0 )); then
    if [[ $1 == 'check' ]]; then
        check_only=1
        shift
    fi
fi
if (( $# > 0 )); then
    while (( $# > 0 )); do
        case "${1##*.}" in
            "c" | "h")
                C_FILES[${#C_FILES[@]}]="$1"
                ;;
            "F90")
                FORTRAN_FILES[${#FORTRAN_FILES[@]}]="$1"
                ;;
            "sh")
                SH_FILES[${#SH_FILES[@]}]="$1"
                ;;
            *)
                echo "unknown suffix"
                ;;
        esac
        shift
    done
else
    readarray -t SH_FILES < <(git ls-files -- '*.sh')
    readarray -t C_FILES < <(git ls-files -- '*.c' '*.h')
    readarray -t FORTRAN_FILES < <(git ls-files -- '*.F90')
fi

success=0
declare -a FAILED_FILES=()

for file in "${SH_FILES[@]}"; do
    if ! bashate --ignore E006 "${file}"; then
        success=1
        FAILED_FILES[${#FAILED_FILES[@]}]=${file}
    fi
done

for file in "${C_FILES[@]}"; do
    # Do not indent typed sources. indent-2.2.12 has a regression
    # https://lists.gnu.org/archive/html/bug-indent/2023-04/msg00000.html and
    # incorrectly aligns the indirection operator `*` in some cases.
    if [[ ${file} =~ typed.c ]]; then
        continue
    fi
    indent ${INDENT_ARGS} "${file}" -o "${file}.indented"
    sed -i -e 's:\s\+$::' "${file}.indented"
    if (( $(diff --brief "${file}" "${file}.indented" | wc -l) > 0 )); then
        success=1
        FAILED_FILES[${#FAILED_FILES[@]}]=${file}
    else
        rm -f "${file}.indented"
    fi
done

for file in "${FORTRAN_FILES[@]}"; do
    ${EMACS} --batch \
        "${file}" \
        --eval "(whitespace-cleanup)" \
        --eval "(indent-region (minibuffer-prompt-end) (point-max) nil)" \
        --eval "(write-file (concat (buffer-name) \".indented\"))"
    if (( $(diff --brief "${file}" "${file}.indented" | wc -l) > 0 )); then
        success=1
        FAILED_FILES[${#FAILED_FILES[@]}]=${file}
    else
        rm -f "${file}.indented"
    fi
done

if (( ${#FAILED_FILES[@]} > 0 )); then
    echo
    echo "Please inspect the following files for linting issues:"
    echo
    for file in "${FAILED_FILES[@]}"; do
        if [[ ${file} =~ .sh$ ]]; then
            echo "bashate ${file}"
            bashate "${file}"
            if (( check_only == 0 )); then
                mv "${file}" "${file}.backup"
                bashate "${file}.backup" > "${file}"
            fi
        else
            echo "diff -Naur ${file} ${file}.indented"
            diff -Naur "${file}" "${file}".indented
            if (( check_only == 0 )); then
                mv "${file}.indented" "${file}"
            fi
        fi
    done
    echo
fi

exit ${success}
