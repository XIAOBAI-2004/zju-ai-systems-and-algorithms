# 金融异常检测

本题使用 DGraph-Fin 联系人图识别欺诈用户。模型为每个目标节点输出正常与欺诈两类概率，实验以 ROC-AUC 比较排序效果。

课程题目见 [作业说明](course_original/作业说明.pdf)、[原题 Notebook](course_original/main.ipynb)和[测试提交指南](course_original/测试提交指南.ipynb)。实验过程和最终选择见[程序报告](程序报告.pdf)。

| 实验 | 方法 | 完整验证 AUC | 已训练模型 |
| --- | --- | ---: | --- |
| 原始 MLP | 20 维节点属性，128 单元隐藏层 | 0.71151 | [mlp20.pt](models/mlp20.pt) |
| 固定 50 个特征 | 20 维节点属性加 30 维有向联系与时间统计，线性模型 | 0.79307 | [linear50.pt](models/linear50.pt) |
| 固定 50 个特征 | 同样的输入，64 单元隐藏层 MLP | 0.80306 | [mlp50.pt](models/mlp50.pt) |
| 固定特征，换模型 | 分箱因子分解机 | 0.80646 | [fm50.pt](models/fm50.pt) |
| 固定特征，换模型 | CatBoost | 0.81043 | [catboost50.pt](models/catboost50.pt) |
| 固定特征，换模型 | LightGBM | 0.81241 | [lightgbm50.pt](models/lightgbm50.pt) |
| 丰富图特征与五模型组合 | 邻居、两跳关系、活动时间及按折计算的训练标签统计；4 个 LightGBM 加 1 个 CatBoost | **0.84027** | [ensemble.pt](models/ensemble.pt) |

五模型组合在 Mo 第 3 次系统测试中的 AUC 为 **0.858**，耗时 4 秒。 表中 0.84027 是原提交版本的完整验证结果；用打包权重重新推理实测为 0.84025。验证和平台测试分别使用各自的数据划分。[详细指标](models/metrics.json) · [成员配置与融合权重](results/optimized_recipe.json)

## 复算验证 AUC

使用 Python 3.11，在本目录安装依赖。原题 Notebook 另外需要与 PyTorch 版本匹配的 torch-sparse 和 torch-scatter，运行训练及复算脚本使用下列依赖即可。

~~~bash
python -m pip install -r requirements.txt
~~~

从 Mo 课程工作区取得课程原始数据，将 dgraph.npz 放在以下路径。数据和逐节点预测缓存保存在本地。

~~~text
datasets/632d74d4e2843a53167ee9a1-momodel/DGraph/raw/dgraph.npz
~~~

~~~bash
python evaluate.py
python evaluate.py --model lightgbm50
~~~

数据已有副本时，用 --data-dir 指定包含 DGraph/raw/dgraph.npz 的上级目录。五模型组合会逐一重建成员所需的图特征，再用已训练的树预测和融合。

## 重新训练

以下命令使用上述默认数据路径，生成的模型保存在 results 下。

~~~bash
python train.py --epochs 200 --lr 0.01 --seed 42
python train_simple.py --hidden 0 --lr 0.003 --epochs 100 --seed 42
python train_simple.py --hidden 64 --lr 0.003 --epochs 100 --seed 42
python train_fm.py --rank 4 --bins 16 --lr 0.003 --epochs 30 --seed 42
python train_shallow_tree.py --engine catboost --depth 3 --trees 2200 --seed 42
python train_shallow_tree.py --engine lightgbm --depth 4 --trees 1400 --seed 42
python train.py --rebuild-best
~~~

## 生成 Mo 提交文件

~~~bash
python evaluate.py --model ensemble --export-mo
~~~

将生成的 results/optimized_model.pt 放到工作区的 results 目录，使用本目录的 [main.py](main.py) 作为预测入口。该入口按课程固定节点编号读取预先计算的预测。报告 PDF 和 LaTeX 源文件位于本目录。
