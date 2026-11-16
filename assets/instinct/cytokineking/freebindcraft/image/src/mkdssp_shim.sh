#!/bin/sh
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
#
# Thin wrapper installed as `mkdssp` in front of the real conda-forge mkdssp (renamed mkdssp.real).
# mkdssp 4.x (libcifpp's PDB reader) aborts with "Error parsing PDB at line 1 / Not a valid integer in
# PDB record" on legacy PDB files that carry free-text REMARK records (e.g. REMARK   Copyright ...), which
# the bundled classic dssp 2.0.4 ignored. Biopython then reports "does not seem to be an mmCIF file" and
# FreeBindCraft silently falls back to default secondary-structure values (0 helix / 0 sheet / 100 loop).
# This wrapper rewrites the PDB input before calling the real binary: REMARK records without a numeric
# remark number are dropped; every other record is kept. All other arguments and the exit status are
# passed through unchanged; mmCIF inputs and `--version` are not touched.
set -u

here=$(dirname "$0")
real=${MKDSSP_REAL:-$here/mkdssp.real}
[ -x "$real" ] || { echo "mkdssp wrapper: real binary $real not found" >&2; exit 127; }

# The input is the last argument that names an existing regular file (Biopython: `mkdssp [flags] in out`
# or `mkdssp -i in -o out`; the output file does not exist yet, so it is never mistaken for the input).
idx=0
inidx=0
for a in "$@"; do
    idx=$((idx + 1))
    case "$a" in -*) continue ;; esac
    [ -f "$a" ] && inidx=$idx
done

if [ "$inidx" -eq 0 ]; then
    exec "$real" "$@"
fi

# Rotate the argument list once, substituting a sanitised copy for the input file.
tmp=$(mktemp "${TMPDIR:-/tmp}/mkdssp-in.XXXXXX") || exec "$real" "$@"
trap 'rm -f "$tmp"' EXIT INT TERM
n=$#
i=0
in=
while [ "$i" -lt "$n" ]; do
    i=$((i + 1))
    a=$1
    shift
    if [ "$i" -eq "$inidx" ]; then
        in=$a
        set -- "$@" "$tmp"
    else
        set -- "$@" "$a"
    fi
done

# mmCIF (or anything that is not a legacy PDB) is passed through as is.
if head -c 5 "$in" | grep -q '^data_'; then
    cp "$in" "$tmp"
    "$real" "$@"
    exit $?
fi
# Free-text REMARK records are the only thing libcifpp cannot parse (a missing HEADER is fine); keep the rest.
awk '!(/^REMARK/ && $0 !~ /^REMARK +[0-9]+( |$)/)' "$in" > "$tmp"
"$real" "$@"
exit $?
