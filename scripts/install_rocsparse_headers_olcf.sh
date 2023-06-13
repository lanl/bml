#!/bin/bash
#
this_dir=$(pwd)
if [[ ! -e src/C-interface || $this_dir == *build ]]; then
    echo "Please run this script in the bml/ root directory"
    exit 1
fi
cp -r $ROCM_PATH/include/rocsparse src/C-interface/.
sed -i -e 's/\[\[.*//' src/C-interface/rocsparse/rocsparse-functions.h
sed -i -e 's/.*\]\] ROCSPARSE_EXPORT/ROCSPARSE_EXPORT/' src/C-interface/rocsparse/rocsparse-functions.h
