from pprint import pprint
from litie.pipelines import RelationExtractionPipeline

"""
litie 的 gplinker 模型，一开始认为效果很差，后来发现其实和mrebel差不多。()
"""

pipeline = RelationExtractionPipeline("gplinker", model_name_or_path="xusenlin/duie-gplinker", model_type="bert")
text = "查尔斯·阿兰基斯（Charles Aránguiz）是智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部。"
pprint(pipeline(text))

