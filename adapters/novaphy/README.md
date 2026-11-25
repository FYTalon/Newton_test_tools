# 评测程序接入说明

评测器通过外部命令调用候选程序。最终交付可以是可执行程序；程序按统一请求运行场景，并将结果写入指定目录。

## 调用接口

```sh
/path/to/candidate --request /path/to/request.json --output /path/to/output
```

`--request` 指定评测器生成的 JSON 请求，包含场景、测试模式、帧数、子步数、运行开关及参考标识。
`--output` 指定本轮结果目录。程序成功完成时返回 0；配置不支持、步进不完整、数据无效或缓冲溢出时返回非零。

使用 `--candidate` 提供候选程序命令时，评测器会自动追加这两个参数。程序需要的运行库和场景资产应随交付包提供。

## 场景范围

下表列出标准要求的范围，接入完成情况以实际运行和复核结果为准。

| 类别 | 场景 |
| --- | --- |
| 刚体与铰链体 | 六个常规场景 |
| 大规模仿真 | 刚体下落、机器狗群组 |
| 流体与流固耦合 | 溃坝、溃坝耦合、TG 涡 |

共 11 个物理场景；流固耦合的两种实现分别测试，共 12 组。场景资产、初态、规模、物理参数和时间步按标准用例执行。
扩展清单保留公共清单中的物理参数、运行长度和轮次，声明实际使用的求解器及碰撞实现标识。

## 运行与复核

将命令变量替换为实际交付的程序命令，从评测工具根目录执行：

使用本仓库的 Python 适配器时，候选命令为
`python -m newton_test_tools worker --adapter adapters/novaphy/adapter.py`，
在可导入 NovaPhy 及所需场景和资产的环境中执行。

```sh
CANDIDATE_COMMAND=/path/to/candidate
REFERENCE_COMMAND=/path/to/reference

newton-test run --suite cases/regular6.json \
  --candidate "$CANDIDATE_COMMAND" --reference "$REFERENCE_COMMAND" \
  --mode all --output runs/regular

newton-test run --suite cases/novaphy_extended.json \
  --candidate "$CANDIDATE_COMMAND" --reference "$REFERENCE_COMMAND" \
  --mode all --output runs/extended

newton-test verify runs/regular
newton-test verify runs/extended
```

正式测试使用清单规定的完整帧数与独立轮次。缩短帧数的运行仅用于检查接入流程。

## 输入输出约定

- 精度轮输出初态、逐帧或逐子步状态，以及该用例要求的接触、冲量、压力、体积或动能观测。
- 流固耦合保留逐帧粒子状态、立方体位姿和速度，并记录状态坐标系及坐标转换。
- 性能轮执行完整的场景步进；计时和显存监测按请求执行。
- 保留运行配置、实际完成帧数、运行日志和原始观测。公共评测工具负责比较、汇总和复核。
- 缺少数据时明确报错，不以空值、零值或通过结论替代。

请求字段和结果格式见[适配器协议](../../docs/adapter.md)，场景与运行长度见[常规清单](../../cases/regular6.json)和[扩展清单](../../cases/novaphy_extended.json)。
