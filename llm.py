import torch
import torch.nn as nn
import torch.optim as optim


# シンプルなLLMの定義
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, padding_idx):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_size, padding_idx=padding_idx)  # パディングIDを追加
        self.fc = nn.Linear(embed_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # 単語埋め込みの平均を取る
        x = self.fc(x)
        return x

# ハイパーパラメータの設定
vocab_size = 10  # 語彙サイズ（「あなた」「誰」「ですか」「こんにちは」「お元気ですか」「はい」「いいえ」「何」「何歳」など）
embed_size = 5  # 埋め込み次元数
output_size = 4  # 出力サイズ（「松田」「田中」「元気です」「こんにちは」）
padding_idx = vocab_size  # パディングID

# モデルの作成
model = SimpleLLM(vocab_size, embed_size, output_size, padding_idx)

# モデルのパラメータ数を確認
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')

# トレーニングデータの準備（単語をIDにエンコード）
word_to_id = {'あなた': 0, '誰': 1, 'ですか': 2, 'こんにちは': 3, 'お元気ですか': 4, 'はい': 5, 'いいえ': 6, '何': 7, '何歳': 8}
id_to_word = {v: k for k, v in word_to_id.items()}

# データをパディングして同じ長さにする
max_length = 3
input_data = [
    [0, 1, 2],  # あなたは誰ですか？
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2],
    [3, padding_idx, padding_idx],        # こんにちは
    [3, padding_idx, padding_idx],
    [3, padding_idx, padding_idx],
    [3, padding_idx, padding_idx],
    [3, padding_idx, padding_idx],
    [4, padding_idx, padding_idx],        # お元気ですか？
    [4, padding_idx, padding_idx],
    [4, padding_idx, padding_idx],
    [4, padding_idx, padding_idx],
    [4, padding_idx, padding_idx],
    [7, 8, padding_idx],                  # 何歳ですか？
    [7, 8, padding_idx],
    [7, 8, padding_idx],
    [7, 8, padding_idx],
    [7, 8, padding_idx]
]
input_data = torch.tensor(input_data, dtype=torch.long)

# ラベルの準備
labels = torch.tensor([0] * 5 + [3] * 5 + [2] * 5 + [1] * 5, dtype=torch.long)  # 「松田」（=0）、「こんにちは」（=3）、「元気です」（=2）、「田中」（=1）

# 損失関数とオプティマイザの定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 簡単なトレーニングループ
for epoch in range(200000):  # エポック数を増やして収束を図る
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:  # 10エポックごとに損失を表示
        print(f'Epoch [{epoch+1}/200000], Loss: {loss.item():.4f}')

# テストデータ
test_questions = [
    [0, 1, 2],               # あなたは誰ですか？
    [3, padding_idx, padding_idx],   # こんにちは
    [4, padding_idx, padding_idx],   # お元気ですか？
    [7, 8, padding_idx]              # 何歳ですか？
]
test_questions = torch.tensor(test_questions, dtype=torch.long)

# CLIでユーザ入力を受け付ける関数
def cli():
    model.eval()
    with torch.no_grad():
        while True:
            user_input = input("質問をどうぞ: ")
            if user_input in ["exit", "quit", "q"]:
                break
            # 入力をトークンIDに変換
            question_tokens = [word_to_id.get(word, padding_idx) for word in user_input.split()]
            # パディングして長さを揃える
            question_tokens = question_tokens + [padding_idx] * (max_length - len(question_tokens))
            question_tensor = torch.tensor([question_tokens], dtype=torch.long)
            # モデルで予測
            test_output = model(question_tensor)
            _, predicted = torch.max(test_output, 1)
            if predicted.item() == 0:
                print("松田")
            elif predicted.item() == 1:
                print("田中")
            elif predicted.item() == 2:
                print("元気です")
            elif predicted.item() == 3:
                print("こんにちは")
            else:
                print("不明な回答")

if __name__ == "__main__":
    cli()
