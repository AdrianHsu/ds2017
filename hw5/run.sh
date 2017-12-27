make
for (( i=6; $i<=10; i=$i+1))
do
    y=$((i-5))
    c="0.000${i}"
    if [ "$i" = "10" ]
    then
        c="0.001"
    fi
    ./fim.out retail.txt ${c} result/${y}${1}${2}.txt
done

