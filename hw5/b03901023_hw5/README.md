# Frequent itemset mining with CUDA (Eclat Algorithm)

1. Implement a GPU version of Eclat algorithm using vertical bit vectors.
2. The output format, for example:
```1 40 (125)```

# How-to
Execute the executable with ```executable_name data_file min_sup out_file```   

for example, ```./fim.out retail.txt 0.001 outfile```
   
# Description
Each line represents one frequent itemset. The numbers in each line separated by a space is the items in the itemset. The number in the parentheses is the support of that itemset.

# Implementation
1. Use shared memory to store the intermediate result (ex: support value). 
2. Use reduction technique to sum the support fast.

