flowchart TD

%% ==========================
%% Title
%% ==========================

A["IE_MOEC 算法流程图<br/>
Independently Evolving Multi-Objective Extremum Combination"]

%% ==========================
%% Stage 1
%% ==========================

B["阶段① 初始化起源种群<br/><br/>
• 随机生成 N 个个体 (N=POPSIZE)<br/>
• 目标函数评估<br/>
• 非支配排序<br/>
• 初始化理想点 z_ideal"]

A --> B


%% ==========================
%% Outer loop
%% ==========================

C{{"外循环开始<br/>
MAX_OUTER_GENS = 20"}}

B --> C


%% ==========================
%% Stage 2
%% ==========================

subgraph S2["阶段② 孤岛分离与扩增"]

D1["M 个极端岛<br/><br/>
• 每目标维度选择最优<br/>
• focus = 0~M-1<br/>
• Tchebycheff 偏置<br/>
w=[ε,...,1,...,ε]"]

D2["M 个折中岛<br/><br/>
• Dirichlet 随机权重<br/>
• focus=None<br/>
• Tchebycheff 均衡搜索<br/>
w~Dir(α=1)"]

D3["祖先 × K<br/>
K=2M 个孤岛<br/><br/>
高变异率扩增 0.15<br/>
每岛生成 island_popsize 个体"]

D1 --> D3
D2 --> D3

end

C --> S2


%% ==========================
%% Stage 3
%% ==========================

subgraph S3["阶段③ 孤岛独立演化 (μ+μ)"]

E1{"判断阶段<br/>
progress < SWITCH_RATIO(0.4) ?"}

E2["聚合阶段 YES<br/><br/>
ISLAND_GENS_EARLY=30<br/>
Tchebycheff 加权聚合<br/>
极端岛偏向自身目标<br/>
折中岛 Dirichlet 权重"]

E3["Pareto阶段 NO<br/><br/>
ISLAND_GENS_LATE=70<br/>
非支配排序选择<br/>
末前沿方向偏置"]

E4["每代演化<br/><br/>
随机配对<br/>
↓<br/>
SBX交叉<br/>
↓<br/>
多项式变异<br/>
↓<br/>
父子合并(2μ)<br/>
↓<br/>
选择μ个存活"]

E1 -->|YES| E2
E1 -->|NO| E3
E2 --> E4
E3 --> E4

end

S2 --> S3


%% ==========================
%% Stage 4
%% ==========================

subgraph S4["阶段④ 极值重组（多精英）"]

F1["每岛选择3个精英<br/><br/>
① 方向最优<br/>
② 最大拥挤度<br/>
③ 随机前50%"]

F2["K×3 精英池<br/><br/>
两两SBX交叉<br/>
+ 多项式变异"]

F3["混血池生成<br/>
≈ C(K×3,2)×2 子代"]

F1 --> F2 --> F3

end

S3 --> S4


%% ==========================
%% Stage 5
%% ==========================

subgraph S5["阶段⑤ 外层 NSGA-III 筛选"]

G1["起源种群<br/>
+<br/>
混血子代"]

G2["NSGA-III参考向量小生境选择"]

G3["ASF自适应归一化<br/>
↓<br/>
Das-Dennis参考点关联<br/>
↓<br/>
小生境补选"]

G4["产生下一轮起源种群 N"]

G1 --> G2 --> G3 --> G4

end

S4 --> S5


%% ==========================
%% Stage 6
%% ==========================

subgraph S6["阶段⑥ 收敛判定"]

H1{"满足停止条件?<br/><br/>
MAX_OUTER_GENS=20<br/>
或<br/>
IGD连续30代无改善"}

H2["未收敛<br/>
返回阶段②"]

H3["已收敛<br/>
进入阶段⑦"]

H1 -->|否| H2
H1 -->|是| H3

end


S5 --> S6

H2 --> S2


%% ==========================
%% Stage 7
%% ==========================

subgraph S7["阶段⑦ PF扩展"]

I1["起源种群每个体<br/>
双倍变异率产生候选解"]

I2["候选池<br/>
≈ POPSIZE × PF_EXPAND_RATIO(2.0)"]

I3["合并<br/>
↓<br/>
NSGA-III筛选<br/>
↓<br/>
最终POPSIZE个解"]

I1 --> I2 --> I3

end


H3 --> S7


%% ==========================
%% Output
%% ==========================

J["输出最终 Pareto 前沿<br/><br/>
评价指标:<br/>
IGD / GD / HV / SP / ONVG<br/>
+ 可视化"]

S7 --> J



%% ==========================
%% Style
%% ==========================

style A fill:#1f4e79,color:white
style C fill:#ffe599

style B fill:#d9ead3
style S2 fill:#e2f0d9
style S3 fill:#fff2cc
style S4 fill:#eadcf8
style S5 fill:#d9eaf7
style S6 fill:#fce4d6
style S7 fill:#d9ead3
style J fill:#c9daf8