#!/bin/bash
# spreadsheetにコピペ & csvの一部として使いたいためformat

file=$1

cat $file | sed -e 's/"/'\''/g' -e 's/^/"""/' -e 's/$/"""/' -e 's/\\//g'
