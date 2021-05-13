# GraphFrames (for Task1)

## to use in pyspark, run:
`pyspark --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11`

## to use "python task1.py ..." and debugger in vscode:
add this line in env/bin/activate:

`export SPARK_HOME="$VIRTUAL_ENV/lib/python3.6/site-packages/pyspark/"`

then copy all the .jar files downloaded when running "pyspark" to $VIRTUAL_ENV/lib/python3.6/site-packages/pyspark/jars

## Reference:

[python - Unable to run a basic GraphFrames example - Stack Overflow](https://stackoverflow.com/questions/39261370/unable-to-run-a-basic-graphframes-example)

[Piazza @303](https://piazza.com/class/kiqe3knu9qc1ad?cid=303)

# Calculate Modularity in Task2

in calculating the modularity, make sure A, m, ki, kj are calculated on the original graph, only S should change

## Reference:

[Piazza @313](https://piazza.com/class/kiqe3knu9qc1ad?cid=313) there is another good blog in the @313 answer