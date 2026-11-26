# 任务 1—3 评分标准

## 分值

| 任务 | 场景 | 性能比 | 精度或物理性 | 显存占用比 | 满分 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1：常规刚体与铰链体 | 1-1、1-2、1-3、1-5、1-6、1-7 | 15 | 轨迹精度 10 | 5 | 30 |
| 2：十万自由度刚体与铰链体 | 1-4、1-8 | 5 | 守恒与接触物理性 3 | 2 | 10 |
| 3：流体与流固耦合 | 1-9、1-10、1-11 | 5 | 流体精度 3 | 2 | 10 |

任务内场景等权。流固耦合两个实现合为场景 1-10。任务 1—3 共 11 个场景，合计满分 50。

## 性能比

每端独立运行 3 次，记录正式运行的帧数 `N_r` 与 GPU 同步计时秒数 `T_r`。

```text
FPS = sum(N_r) / sum(T_r)
x = FPS_candidate / FPS_reference
H = max(x_team_1, x_team_2)

S_fps = 80*x                             x <= 1
S_fps = 80 + 20*(x - 1)/(H - 1)          x > 1
```

1-1 至 1-10 以 Newton 为基准；1-11 的基准为 30 FPS。1-10 的候选 FPS 为两个流体版本各自三轮汇总 FPS 的算术平均。计时排除编译、加载、初始化、预热、渲染及帧率限制，覆盖完整物理帧和全部子步。

## 显存占用比

```text
M = (M_peak_1 + M_peak_2 + M_peak_3) / 3
u = M / M_GPU
x = u_candidate / u_reference

S_mem = 100                              x <= 0.8
S_mem = 100*(2.0 - x)/1.2                0.8 < x < 2.0
S_mem = 0                                x >= 2.0
```

峰值覆盖全部正式物理帧中进程持有的求解缓冲、初始化分配与缓存。1-11 的参考峰值为 6144 MiB，即 `6442450944` 字节；其余场景采用 Newton 三次运行峰值的均值。

## 任务 1：轨迹精度

```text
x = max(x_C, x_D, x_K)
```

| 指标 | 计算 |
| --- | --- |
| `x_C` | 每帧有效接触形状对的对称差数量除以并集数量，取全轨迹最大值；双方均无接触记 0 |
| `x_D` | 全部动态刚体的合力、质心合力矩归一化误差最大值；分母固定为 Newton 全轨迹峰值与场景正下限的较大者 |
| `x_K` | 质心位置、姿态、线速度、角速度归一化误差最大值；尺度为固定的 `L`、`pi`、`v_ref`、`omega_ref` |

姿态误差为 `2*acos(abs(dot(q_candidate, q_reference)))/pi`。

```text
S_accuracy = 100                         x <= 0.01
S_accuracy = 100 - 20*(x - 0.01)/0.09    0.01 < x <= 0.10
S_accuracy = 80*(0.50 - x)/0.40          0.10 < x < 0.50
S_accuracy = 0                           x >= 0.50
```

合格线为 `x<=0.10`。保存第 0 至 N 帧状态与第 1 至 N 帧有效接触。

## 任务 2：守恒与接触物理性

有效自由度至少 100000，物理检查完成 500 帧及全部子步。

```text
R_P = max_t(norm(P(t) - P(0))) / P_ref
R_E = max_t(abs(E(t) - E(0))) / E_ref
delta_mean = sum(max(0, -contact_distance)) / (L * penetrating_sample_count)
F_negative = max(max(0, -normal_force)) / F_ref
```

`P(t)` 为所有世界的总线动量，`E(t)=K(t)+U(t)`，不含软接触内部储能。`P_ref`、`E_ref`、`L`、`F_ref` 在测试前固定。

排除 Newton 绝对值不大于 `0.001` 的分量，其余分量分别计算 `r=value_candidate/value_reference`：

```text
S_component = 100                        r <= 0.95
S_component = 100 - 20*(r - 0.95)/0.10   0.95 < r <= 1.05
S_component = 80*(1.20 - r)/0.15         1.05 < r < 1.20
S_component = 0                          r >= 1.20

S_physics = mean(S_component)
```

合格线为各有效分量 `r<=1.05`。同时记录有效自由度、逐子步状态与接触、总质量、质心、世界惯性、线动量、质心角动量、K/U/E 及独立碰撞事件。

## 任务 3：流体精度

### 1-9 纯流体溃坝

粒子位置与线速度分别按固定尺度归一化，在全部粒子和规定帧中取最大误差，再取两项最大值得到 `x_F`。使用任务 1 的 `1%→100、10%→80、50%→0` 分段计分。

### 1-10 流固耦合

| 分量 | 实现 | 满分误差 | 零分误差 | 权重 |
| --- | --- | ---: | ---: | ---: |
| 底面压强分布 | V1 | 5% | 100% | 25% |
| 立方体状态 | V1 | 5% | 100% | 25% |
| 体积损失 | V2 | 10% | 100% | 50% |

```text
S_k = max(0, min(100, 100*(1 - E_k)/(1 - E_k0)))
S_coupled = 0.25*S_pressure + 0.25*S_cube + 0.50*S_volume
```

压强采用 `32×32` 底面采样单元的法向接触冲量和边界速度修正冲量；体积采用统一 CIC 有效体积曲线。

### 1-11 TG 涡

以公开谱数据为参照，采样时间为 `t=0,0.01,…,19.99 s`。

```text
E_K = sqrt(sum((K - K_reference)^2) / sum(K_reference^2))
E_dissipation = sqrt(sum((epsilon - epsilon_reference)^2) / sum(epsilon_reference^2))
E = (E_K + E_dissipation) / 2

S_TG = 100                               E <= 0.03
S_TG = 100*(0.40 - E)/0.37               0.03 < E < 0.40
S_TG = 0                                 E >= 0.40
```

## 运行配置

| 场景 | 精度次数 | 精度确定性 | 精度 Graph | 性能次数 | 性能确定性 | 性能 Graph | 单场景纯测试时限 |
| --- | ---: | --- | --- | ---: | --- | --- | --- |
| 任务 1 六例 | 1 | 开 | 关 | 3 | 关 | 开 | 15 分钟 |
| 1-4 | 1 | 关 | 关 | 3 | 关 | 开 | 30 分钟 |
| 1-8 | 3 | 关 | 关 | 3 | 关 | 开 | 30 分钟 |
| 1-9 | 3 | 关 | 场景配置 | 3 | 关 | 场景配置 | 15 分钟 |
| 1-10 各实现 | 3 | 开 | 场景配置 | 3 | 关 | 场景配置 | 15 分钟 |
| 1-11 | 3 | 关 | 场景配置 | 3 | 关 | 场景配置 | 15 分钟 |

## 分数汇总

观测点映射分 `S` 的范围为 0—100，观测点分值为 `P`。

```text
场景分 = sum(P * S / 100)
任务分 = mean(该任务各场景分)
总分 = 任务1分 + 任务2分 + 任务3分
```

中间值不舍入，任务分和总分四舍五入保留两位小数。缺失或失效的观测点记 0；有效自由度不足 100000 的任务 2 场景各项记 0。崩溃、卡死、发散、显存不足或超时按相关观测点记 0。

## 计算命令与输入

从评测结果目录读取数据：

```sh
newton-test score-reports --team team_a runs/team-a-regular --team team_a runs/team-a-extended \
  --team team_b runs/team-b-regular --team team_b runs/team-b-extended --output scores.json
```

同一团队可指定多个结果目录；每个场景只能提供一份结果。输出保留源报告位置、计时与显存的逐轮信息和失败原因。运行帧数、轮次或确定性开关与场景配置不符时，对应得分为待定。常规刚体与十万自由度场景还校验 CUDA Graph 模式。缺少另一家结果时，依赖 `H` 的性能分为待定。

流固耦合的性能比按两个版本的汇总 FPS 取均值。输出分别保留各轮的最大立方体中心相对误差、最大底面压强相对误差、最大体积比例差，以及缺失帧和湿区差异。精度和显存的跨版本、跨轮汇总在评分口径确定后计算，当前显示待定。

也可从测量数据文件计算：

```sh
newton-test score --input measurements.json --output scores.json
```

规则文件为 [cases/scoring.json](../cases/scoring.json)。输入顶层为 `teams` 数组，包含两个对象；每个对象提供 `id` 和以文档场景编号为键的 `scenes`。每个场景提供下列字段。

| 字段 | 内容 |
| --- | --- |
| `fps_ratio` | 三次正式运行汇总得到的性能比 |
| `gpu_peak_ratio` | 三次正式进程峰值显存均值的占用率之比 |
| `accuracy.x_C/x_D/x_K` | 任务 1 全轨迹误差 |
| `effective_dof` | 任务 2 实际有效自由度 |
| `completed_frames/completed_substeps` | 任务 2 物理检查帧数及子步数 |
| `accuracy.candidate/reference` | 任务 2 四项指标，键为 `R_P_max`、`R_E_max`、`mean_penetration_over_length`、`negative_normal_peak_over_force` |
| `accuracy.x_F` | 1-9 全粒子全轨迹最大归一化误差 |
| `accuracy.pressure_error/cube_state_error/volume_loss_error` | 1-10 三项归一化误差 |
| `accuracy.energy_relative_l2/dissipation_relative_l2` | 1-11 两条曲线的相对 L2 误差 |
| `status` | 正常运行填 `ok`；失效填失败状态 |
| `failed_observations` | 失效观测点名称数组：`performance`、`accuracy`、`memory` |

输出包含每个观测点的原始值、锚点、映射分、折算分，以及每个任务得分和 50 分制总分。
