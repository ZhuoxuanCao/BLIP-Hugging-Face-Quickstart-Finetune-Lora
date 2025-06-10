import json

# 输入你的 JSON 文件名和输出的 JSONL 文件名
input_json = "./annotations.json"      # 你的当前文件（实际上是JSON数组）
output_jsonl = "./annotations.jsonl.out"  # 输出文件（jsonl格式）

# 读取整个数组
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# 写入 jsonl 格式（每行为一个 JSON 对象）
with open(output_jsonl, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("转换完成！输出文件：", output_jsonl)
