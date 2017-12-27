a=$(ls result)

for i in $a; do
    a=$(tail -n 1 result/$i)
    echo $a $i
done
