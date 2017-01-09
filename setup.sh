#!/bin/sh
set +x

# Get data from Udacity
wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

# Unzip it
unzip data.zip

# Cleanup
rm -rf data.zip
rm -rf __MACOSX

# Move to current folder 
mv data/ udacity_data/
