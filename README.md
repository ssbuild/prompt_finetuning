## 安装

- pip install -U deep_training >= 0.0.16
- 当前文档版本pypi 0.0.16

## 更新详情

- [deep_training](https://github.com/ssbuild/deep_training)

## 深度学习常规任务例子

- [deep_training-pytorch-example](https://github.com/ssbuild/deep_training-pytorch-example)
- [deep_training-tf-example](https://github.com/ssbuild/deep_training-tf-example)

## clue-prompt finetuning 

    预训练模型下载 https://huggingface.co/ClueAI/PromptCLUE-base-v1-5

## 数据示例

{"input": "我可以用以下的句子：“花呗在什么时间段可以用”，来替换这个句子：“什么时候用了花贝”，并且它们有相同的意思？。选项：是的，不是。答案：", "target": "不是", "type": "classify"}
{"input": "摘要：针对水平受荷桩在不同的长径比和桩土刚度比条件下可以表现出刚性桩、半刚性桩或柔性桩变形特性的特点,运用刚性桩和柔性桩的临界桩长计算公式,结合相似原理,推导了重力加速度为1g条件下缩尺模型桩与原型桩的临界桩长相似比,与几何相似比进行对比,评判模型桩和原型桩的变形特性,分析桩身材料模量相似比与几何相似比对模型桩和原型桩变形特性相似性的影响,并通过有限元方法进行数值模拟验证.研究结果表明:桩身材料模量是控制模型桩与原型桩满足变形特性相似的主要参数；直接采用原型桩材和原型土开展的模型试验与原型桩的变形特性不具有相似性,但通过选择模量相似比接近于几何相似比的模型桩材可以使得模型试验结果与原型相似.\n 以下的关键词都是这篇摘要合适的关键词吗？关键词：几何，模型试验，特性，相似性。答案是：\n选项：是的，不是\n答案：", "target": "不是", "type": "classify"}
{"input": "下面两个句子语义是“相同”或“不同”？“我买商品怎么用不了花呗”，“我的花呗怎么用不了”。选项：相同，不同。答案：", "target": "不同", "type": "classify"}



# 使用方法

## 生成训练record

    python data_utils.py
    
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)

## 训练

python task_prompt_t5.py
