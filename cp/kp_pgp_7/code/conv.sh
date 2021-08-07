i=0
while [[ $i != 30 ]]; do
    python conv.py out/$i.data out_png/$i.png
    i=$(($i+1))
done