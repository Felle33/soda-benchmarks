#!/usr/bin/env bash
# The legacy launch host is valid only for the whole-kernel pipeline.
if [[ ${1:?compilation mode} != kernel ]]; then
  echo 'Operator mode must use sodap-tosa-network-host; refusing sodap-launches-to-esp' >&2
  exit 1
fi
