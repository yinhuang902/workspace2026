
填表，要用snog，LG，simplex方法run 1%, 0.1%, 0.01%

snoglode-main-2-ori-rundata
先改造成run serially并且记录数据跑起来
现在home pc上用1e-2 serially 50scen运行，同时改了Kd的bound


pid_simplex_mixmscs_time_moreub-serially
每一轮不用重新搞simplex LB了，而且试了5分钟10scen 1e-1结果一样
要改成run serially的形式

pid_simplex_mixmscs_time_moreub-serially - 副本
每一轮不用重新搞simplex LB了，而且试了5分钟10scen 1e-1结果一样
还没有改run serially的形式，先保存一下


snoglode-main-2-ori-record3v5
可以在1e-3运行（修改了lbub在obbt以后大小相反的问题，应该是对的但不百分比确定，运行起来没有明显问题）

-----------------------data-----------------------------
sp_snog_result_1e2
home pc上用1e-2 serially 50scen运行，同时改了Kd的bound

sp_snog_result_1e3
home pc上用1e-3运行，用snoglode-main-2-ori-record3v5

