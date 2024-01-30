# ------------------------------------------------------------------------------
#  Parameters
# ------------------------------------------------------------------------------
name=glove300
n=2196017  #data size
d=300      #dimension
qn=1000    #query size
k=100      #topk

efc=2000   #HNSW parameter
M=32       #HNSW parameter
L=15       #level 
eps=0.2    #epsilon
Tn=100    #size of quantile table

dPath=./${name}/${name}_base.fvecs   #data path
qPath=./${name}/${name}_query.fvecs  #query path
tPath=./${name}/${name}_truth.ivecs        #groundtruth path

#----Indexing for the first execution and searching for the following executions---------

./build/PEOs ${dPath} ${qPath} ${tPath} ${n} ${qn} ${d} ${k} ${efc} ${M} ${L} ${eps} ${Tn}

