# 日経レバ1泊トレード判定

前日の大引けで1570を買い、翌営業日の寄り付きで売る戦略を検証するStreamlitアプリです。

## ファイル
- A_app.py
- A_requirements.txt
- A_README.md

## 今回の修正
「本日の数値を条件にコピー」ボタンで、当日のNASDAQ100先物・SOX・ドル円・VIXの実測値をバックテスト条件へコピーできます。

Streamlitで問題になっていた「ウィジェット生成後にst.session_stateを書き換える」処理を修正し、コピー処理をサイドバーの入力欄生成より前に実行する構造にしています。

## 起動
```bash
pip install -r A_requirements.txt
streamlit run A_app.py
```

## 注意
バックテストは実運用の約定条件を完全には再現しません。手数料、スリッページ、1570の実際の寄り付き価格などは別途考慮してください。
