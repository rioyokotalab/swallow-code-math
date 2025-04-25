import json
import matplotlib.pyplot as plt
import collections

# フォントサイズを全体的に大きめに設定（論文用）
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

# スコアを読み込む（小数点は切り捨て）
scores = []
with open('/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-merged/swallow-code-v0.3-no-repet.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'score' in data:
            score = data['score']
            scores.append(int(score))  # 小数点切り捨て

# スコアの分布を集計
score_counts = collections.Counter(scores)
bins = list(range(0, 11))  # スコア0〜10
frequencies = [score_counts.get(b, 0) for b in bins]
colors = ['gray' if b < 6 else 'orange' for b in bins]

# グラフ描画
plt.figure(figsize=(10, 6))
plt.bar(bins, frequencies, color=colors, edgecolor='black')
plt.axvline(5.5, color='red', linestyle='--', label='Score >= 6 threshold')
plt.xlabel('Pylint Score')
plt.ylabel('Sample Count')
# plt.title('Score Distribution with Highlighted Threshold')
plt.xticks(bins)
plt.legend()
plt.tight_layout()
plt.savefig('score_distribution.png', dpi=300)  # dpiも論文用に高めに設定
