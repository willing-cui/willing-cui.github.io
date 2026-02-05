# Troubleshooting

1. 公式中的换行符 `\\`，因为与转义字符冲突，如果出现公式无法正常换行，考虑全局替换 `\\` 为 `\\\\`。

2. 公式中的 `_`，可能与markdown语法中的斜体冲突，_就像这样_ (`_就像这样_`)，如果出现公式无法正常显示，优先考虑全局替换 `_` 为 `\_`。

3. 公式中的 `\%`，要替换为`\\%`，反斜杠数量加倍。

4. 超链接跳转时，打开新的标签页：

   ```html
   <a href="https://超链接" target="_blank" rel="noopener noreferrer">超链接名称</a>
   ```

5. 标准图片格式：

   ```html
   <span class="image main">
   <img class="main img-in-blog" style="max-width: 50%" src="./blogs/folder_name/img_name.webp" alt="img_name" />
   <i>Caption</i>
   </span> 
   ```