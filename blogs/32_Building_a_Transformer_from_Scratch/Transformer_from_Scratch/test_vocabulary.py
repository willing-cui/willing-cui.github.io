import sentencepiece as spm

# 加载训练好的模型
model_prefix = 'zh_wiki_spm_comprehensive'
sp = spm.SentencePieceProcessor()
sp.load(f"{model_prefix}.model")

# 测试句子
test_sentence = "数学是研究数量、结构以及空间等概念的一门学科。"

# 编码：将句子转换为Token ID列表
ids = sp.encode_as_ids(test_sentence)
print(f"Token IDs: {ids}")

# 编码：将句子转换为Token 字符串列表
pieces = sp.encode_as_pieces(test_sentence)
print(f"Tokens: {pieces}")

# 解码：将Token ID列表转换回句子
decoded = sp.decode_ids(ids)
print(f"Decoded: {decoded}")

# 打印词表大小
vocab_size = sp.get_piece_size()
print(f"词表大小: {vocab_size}")

# 查看词表
vocab_list = []
for i in range(sp.get_piece_size()):
    vocab_list.append((i, sp.id_to_piece(i), sp.get_score(i)))

# 打印前50个词条
print("\n词表前50个条目:")
for i, piece, score in vocab_list[:50]:
    print(f"{i}\t{piece}\t{score}")