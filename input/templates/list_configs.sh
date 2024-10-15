for i in {0..0}; do
    for dir in ./$i/Rep_*/; do
	readlink -f $dir/run1* | sort -V | head -n -1
    done
done
