#!/bin/sh
# Build the demo warehouse on first start (4 MB download, checksummed), then run quarry.
set -eu
if [ ! -f data/warehouse.sqlite ]; then
    quarry demo build
fi
exec quarry "$@"
