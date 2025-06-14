import jieba
import re

def clean_text(text):
    """
    去除文本中的标点符号和特殊字符，只保留中文、英文、数字。
    """
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)

def tokenize(text):
    """
    使用jieba对文本进行分词，返回词语列表。
    """
    return list(jieba.cut(text)) 