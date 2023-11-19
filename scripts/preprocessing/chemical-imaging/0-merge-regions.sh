find ./data/chemical-images/Uploaded/ \
  | sed -n 's/.*Slide \([0-9]\).*Region\([A-Z]\).*\.mat/\1\2/p' \
  | sort | uniq \
  | while read arg; do
    python ./scripts/preprocessing/chemical-imaging/0-merge-regions.py $arg &
  done
