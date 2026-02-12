
pid_simplex_method
加了EF，每轮用每个scenario的一阶段变量输入ipopt做warmstart
第一轮iteration强制用MIPGap=1e-1，后续改回预设值


snoglode-main-2-EFfixed
EF修正了如果gurobi解subproblem如果超时就不把数值喂给ipopt的问题


我想实现一个功能，在用ipopt计算EF问题的时候，如果当前的单形的体积特别小（容差设置为1e-1），在用默认


