import torch
import re
from collections import defaultdict


def save_pred_details(writer, val_labels, val_outputs, num_classes, class_names, sample_list, step):
    val_labels = val_labels.cpu()
    val_outputs = val_outputs.cpu()

    if isinstance(val_outputs, list):
        val_outputs = torch.cat(val_outputs, dim=0)
    if isinstance(val_labels, list):
        val_labels = torch.cat(val_labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if val_labels.ndim == val_outputs.ndim:
        val_labels = torch.argmax(val_labels, dim=-1)
    # Get the predicted class indices for examples.
    val_outputs = torch.flatten(torch.argmax(val_outputs, dim=-1))
    val_labels = torch.flatten(val_labels)

    correctness = (val_outputs == val_labels).int() # 1=正确，0=错误
    nocorrect_list = []
    for i, item in enumerate(correctness):
        if item != 1:
            nocorrect_list.append(sample_list[i].split(' ')[0])
    nocorrect_list.sort()
    
    classified_data = defaultdict(list)
    for item in nocorrect_list:
        match = re.match(r'([a-zA-Z]+)(\d+)', item)
        if match:
            category, number = match.groups()
            if category in class_names:
                classified_data[category].append(item)

    # 确保所有类别都被统计，即使数据里没有
    final_output = {category: classified_data[category] for category in class_names}
    text = f"""### Number of Incorrect Prediction(Number of Test Sets): {len(nocorrect_list)}({len(sample_list)})\n"""
    table = dict_to_markdown_table(final_output)
    text = text + table
    # for key in list(final_output.keys()):
    #     datas = final_output[key]
    #     text = text + f"""#### num of '{key}': {len(datas)}\n"""
    #     if len(datas)==0:
    #         text = text + "\n"
    #     else:
    #         for data in datas:
    #             text = text + f"{data}\n"
    writer.add_text('Prediction_Details', text, global_step=step)

def dict_to_markdown_table(data_dict):
    """
    将字典转换为 Markdown 表格格式，其中 key 作为表头
    :param data_dict: 输入字典，key 为类别，value 为包含样本的列表
    :return: Markdown 格式的表格字符串
    """
    # 获取所有 key 作为表头
    headers = list(data_dict.keys())
    headers_text = headers.copy()

    # 给headers里加上数量
    for i in range(len(headers_text)):
        headers_text[i] = headers_text[i] + f"(num:{len(list(data_dict.values())[i])})"

    # 找到最长的 value 列表，确保所有列对齐
    max_len = max(len(values) for values in data_dict.values())

    # 初始化表格字符串
    markdown_str = "| " + " | ".join(headers_text) + " |\n"
    markdown_str += "| " + " | ".join(["---"] * len(headers_text)) + " |\n"

    # 填充表格内容（按行排列）
    for i in range(max_len):
        row = []
        for key in headers:
            if i < len(data_dict[key]):
                row.append(data_dict[key][i])  # 获取当前 key 下的第 i 个元素
            else:
                row.append("")  # 列表不够长时补空白
        markdown_str += "| " + " | ".join(row) + " |\n"

    return markdown_str
