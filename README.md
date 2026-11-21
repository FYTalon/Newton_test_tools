# Newton_test_tools

公用仿真评测工具。参考引擎和候选引擎各提供一个适配器命令。工具按同一份场景配置运行两端，保留原始数据并计算误差和性能比。各引擎可使用自己的应用、语言和运行环境。运行器和评分器只依赖数据协议。

## 常规六例

场景以双方已有 example 为入口。建模、控制、碰撞与求解顺序由 example 实现。这一版使用 [regular6.json](cases/regular6.json) 作为共用测试清单。

| 场景 ID | 求解器 | world 数 | 每帧子步数 |
| --- | --- | ---: | ---: |
| `basic_shapes` | XPBD | 1 | 5 |
| `basic_conveyor` | XPBD | 1 | 10 |
| `nut_bolt_hydro` | XPBD | 20 | 4 |
| `basic_pendulum` | XPBD | 1 | 10 |
| `robot_g1` | MuJoCo 外部碰撞 | 4 | 6 |
| `robot_anymal_c_walk` | MuJoCo 外部碰撞 | 1 | 4 |

- 精度从相同初态连续跑 500 帧。双方启用确定性并关闭 graph。每端独立运行两次，用第一次轨迹比较，第二次用来记录该端轨迹是否可重复。接触取每个完整帧末碰撞管线留下的有效记录。输出 `x_C`、`x_D` 和 `x_K`。
- 性能在同一张 GPU 上顺序运行。关闭确定性，有 graph 执行模式的适配器启用 graph。每例每端独立运行 3 次，每次预热 5 帧后连续计时至少 600 秒。每 5 帧同步 GPU，每次 FPS 用完整帧数除以实际计时秒数。计时段只有仿真步进和同步。另由外部运行器自启动到退出按 10 ms 间隔用 NVML 轮询目标进程驻留显存，每次独立取采样峰值。每端分别对 3 次 FPS 和显存峰值取算术平均，报告中的比值使用两端均值相除。

参考引擎可以任意指定。使用 Newton 作为参考时，运行器从 Newton 轨迹导出标准刚体与形状标识表，并提供给候选适配器。二者形状的标识、所属刚体和分类要一致。姿态四元数用于质心位置换算和力矩计算。运动学误差沿用现行代码中的位置、线速度和角速度三项。

## 1. 安装和自检

从本仓库根目录安装公用工具和官方 Newton 1.6.0。首次运行会按 Newton example 的配置获取资产。正式测试含 20 world 的 `nut_bolt_hydro` 时，先按说明给 Newton 检出应用随附的容量与排序键补丁。[Newton 适配器安装说明](adapters/newton/README.md)

用 Newton 对 Newton 跑一次短测，确认参考环境、轨迹输出、graph replay 和显存采样。

```sh
NEWTON_ADAPTER=".venv/bin/python adapters/newton/run.py"

uv run --no-sync newton-test run --suite cases/regular6.json --case basic_shapes \
  --reference "$NEWTON_ADAPTER" --candidate "$NEWTON_ADAPTER" \
  --frames 8 --repetitions 1 --duration 2 --performance-repetitions 1 \
  --output runs/newton-smoke

uv run --no-sync newton-test verify runs/newton-smoke
```

## 2. 接入参评引擎

参评引擎提供一个可执行命令，接受运行器追加的 `--request` 和 `--output`。适配器读取场景 ID 和模式开关，调用自己已有的场景构建与步进接口。Python 引擎可在自己的环境安装本工具，复用 `Monitor`。其他语言可以直接按协议写 JSON。[适配器接口和模板](docs/adapter.md)

用候选引擎先替换自检命令中的 `--candidate`，并把输出目录改成 `runs/candidate-smoke`。例如候选适配器命令可写成 `/path/to/engine/.venv/bin/python /path/to/engine/adapter.py`。精度请求中的 `reference_identity` 提供 Newton 的刚体和形状标识，适配器据此映射本地索引。

## 3. 跑正式六例和查看结果

```sh
NEWTON_ADAPTER=".venv/bin/python adapters/newton/run.py"
CANDIDATE_ADAPTER="/path/to/engine/.venv/bin/python /path/to/engine/adapter.py"

uv run --no-sync newton-test run --suite cases/regular6.json --mode all \
  --reference "$NEWTON_ADAPTER" --candidate "$CANDIDATE_ADAPTER" \
  --output runs/regular6

uv run --no-sync newton-test report runs/regular6
uv run --no-sync newton-test verify runs/regular6
```

每次指定新的输出目录。可重复传 `--case` 选择多个场景，用 `--mode accuracy` 或 `--mode performance` 单独运行。实际配置写入 `run.json`。默认监测 NVML 物理设备 0，`--gpu` 可传索引或 GPU UUID。Newton adapter 使用 `cuda:0`。有多张卡时，可设置 `CUDA_VISIBLE_DEVICES` 使所选 GPU 成为两端进程内的 `cuda:0`。

根目录的 `report.md` 汇总结果，`report.json` 保留每帧误差、参考尺度、原始计时、显存和运行配置。每个适配器的请求、轨迹、报告和日志位于 `cases/<场景>/<模式>/`。执行 `verify` 会核对文件哈希，并从保留的轨迹和原始性能计数重算结果。

可以用已有轨迹离线复算精度：

```sh
uv run --no-sync newton-test compare --suite cases/regular6.json --case basic_shapes --frames 8 \
  --reference runs/newton-smoke/cases/basic_shapes/accuracy/reference/run-1/trajectory.json \
  --candidate runs/newton-smoke/cases/basic_shapes/accuracy/candidate/run-1/trajectory.json
```
