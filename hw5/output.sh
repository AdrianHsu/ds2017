a=$(ls result)

IFS=' '
for i in "${a[@]}"; do
    echo i
done
