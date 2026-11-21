# Newton 1.6 参考适配器

`adapter.py` 用官方 Newton example 实现[公用评测协议](../../docs/adapter.md)。参考环境为 Python 3.11、Newton v1.6.0、Warp 1.17.0、MuJoCo 和 MuJoCo Warp 3.12.0，以及 NVIDIA GPU。该适配器可在只安装 Newton 和本工具的环境中运行。

## 安装

从本工具仓库根目录执行。公用工具和 Newton adapter 共用工具目录下的 `.venv`。

```sh
NEWTON_SRC="$PWD/newton-v1.6.0"
git clone --depth 1 --branch v1.6.0 https://github.com/newton-physics/newton.git "$NEWTON_SRC"

uv sync --locked --python 3.11
uv pip install --python .venv/bin/python \
  -e "$NEWTON_SRC" -r adapters/newton/requirements.txt
```

首跑会按 Newton example 的设定下载资产。G1 和 ANYmal 使用 Newton v1.6.0 自带的固定 newton-assets 版本。螺母螺栓网格按原例子从 IsaacGymEnvs 获取。标准源代码见 [Newton v1.6.0](https://github.com/newton-physics/newton/tree/v1.6.0)。

## 自检和接入

用两次 Newton 运行验证工具链和图、显存监测。

```sh
NEWTON_ADAPTER=".venv/bin/python -m newton_test_tools worker --adapter adapters/newton/adapter.py"

uv run --no-sync newton-test run --suite cases/regular6.json --case basic_shapes \
  --reference "$NEWTON_ADAPTER" --candidate "$NEWTON_ADAPTER" \
  --frames 8 --repetitions 1 --duration 2 --performance-repetitions 1 \
  --output runs/newton-smoke

uv run --no-sync newton-test verify runs/newton-smoke
```

正式比较时将 `--candidate` 替换为参评引擎的适配器命令，使用清单的默认测试长度。目录中的 `identity.json` 给出参考引擎的刚体和形状标识。候选端的精度请求也包含同一份映射。

官方 example 没有统一的 graph 和确定性参数。适配器在试验进程内配置初始化，取得传给 `newton.examples.run` 的场景，并把 `step()`、读数接口和物理 graph 句柄交给工具。精度模式设置求解器 `GPU_TO_GPU` 与外部碰撞排序，跳过 example 的 graph 捕获及其内部预热。MuJoCo 的世界质心速度从分离的 FK 状态取得。性能模式使用 example 的原生 graph，公用 `worker` 在首个预热步核对 graph 重放。两种模式均配置 MuJoCo 外部碰撞和关闭传感器。报告记录运行参数、Python 和 Newton 版本以及 example 哈希。

## 20 world Hydro 的确定性参考

官方 v1.6.0 的确定性末级体素上限是 419,430，完整 500 帧可能溢出。既有参考采用 **838,835** 上限，同时将 anchor 排序键移到 `0x7FFF00..0x7FFFFF`。两项需要配套，避免普通接触键与 anchor 键重叠。适配器要求两项配置均符合这份参考，并将容量写入报告。

要跑含该例的正式精度，在独立 Newton v1.6.0 检出上应用随附补丁。使用上面以可编辑模式安装的环境，应用后即可运行。

```sh
git -C "$NEWTON_SRC" apply --check "$PWD/adapters/newton/hydro-v1.6.0.patch"
git -C "$NEWTON_SRC" apply "$PWD/adapters/newton/hydro-v1.6.0.patch"
```

这项结果标注为“Newton v1.6.0 + Hydro 确定性容量补丁”。其余五例及六例的非确定性性能可直接使用官方 v1.6.0。
