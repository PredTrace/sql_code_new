And(,
    
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ) ==
And(
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    t0-tup-1-n1.n_regionkey == rowsel-v-n_regionkey,
    ------ t0-tup-1-n1.n_regionkey == rowsel-v-r_regionkey,
    ,
    ,
    ,
    ,
    t1-tup-1-region.r_regionkey == rowsel-v-n_regionkey,
    )

t1-tup-1-region.r_regionkey == rowsel-v-r_regionkey,
t0-tup-1-n1.n_regionkey == t1-tup-1-region.r_regionkey
t0-tup-1-n2.n_regionkey == rowsel-v-n_regionkey,

=> 



----t0-tup-1-n2.n_regionkey == rowsel-v-23-n_regionkey,
----t1-tup-1-region.r_regionkey == rowsel-v-63-r_regionkey,
----t0-tup-1-n1.n_regionkey == t1-tup-1-region.r_regionkey,

----t0-tup-1-n2.n_regionkey == rowsel-v-23-n_regionkey,
t0-tup-1-n1.n_regionkey == rowsel-v-23-n_regionkey,
t0-tup-1-n1.n_regionkey == rowsel-v-28-r_regionkey,
t0-tup-1-n1.n_regionkey == rowsel-v-63-r_regionkey,
----t1-tup-1-region.r_regionkey == rowsel-v-63-r_regionkey,
t1-tup-1-region.r_regionkey == rowsel-v-23-n_regionkey,