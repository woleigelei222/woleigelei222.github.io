## 先看懂：这是网站，不是在线编辑器

这个起步版把 Markdown 文件转换为网页。访客可以阅读、搜索与筛选，但不能登录后台编辑。

你日常修改的是 `content/` 里的 Markdown 文件。生成后的 `public/` 是展示结果，**不要直接在里面维护正文**，因为下次生成会覆盖它。

## 第一次使用

先在 `site.yml` 修改姓名和简介，再在 `pages/about.md` 写个人介绍。没有公开联系方式时，保持相关设置为空。

四篇演示文章都明确标为示例，不计入正式记录。准备对外分享前，删除 `content/example-*.md`，或把 `site.yml` 的 `show_examples` 改为 `false`。隐藏示例并不会删除公开仓库中的源文件。

## 每天怎样新增记录？

从下面选择一份模板，复制到 `content/`，取一个便于识别的英文文件名。修改标题、日期、摘要与正文，并为 `slug` 设置一个不重复的英文短名。

[论文笔记模板](templates/paper.md) · [算法理解模板](templates/algorithm.md) · [项目实践模板](templates/project.md) · [学习日志模板](templates/journal.md)

`kind` 只使用 `paper`、`algorithm`、`project` 或 `journal`。`tags` 是主题标签。`featured: true` 表示在首页精选区展示；首版最多显示三篇。

确认可公开后，把 `published` 改为 `true`。未完成的模板默认是 `false`。

## Markdown 最小写法

```markdown
## 一个小标题

这里写自己的解释。**这是重点**。

- 一个观察
- 一个仍然不确定的问题

原文链接写法：[论文页面](https://example.com/)

行内变量可以写成 `latency_ms`。
```

支持常见的标题、列表、链接、表格、代码块与脚注。当前起步版没有接入 LaTeX 公式渲染；公式可以先用代码块准确记录，不要假定美元符号中的内容会自动排版。HTML 标签会转义显示，不会作为脚本执行。

## 本地生成：可选路线

在项目目录安装 `requirements.txt` 中的依赖，然后运行：

```text
python build.py
```

打开 `public/index.html` 即可查看。生成时会检查元数据、重复短名与站内文件链接；有错误时会给出提示。

Windows 的具体命令与 GitHub 发布步骤写在项目根目录的 `README.md`。下载压缩包中的 `public/` 已经生成好，不需要安装环境也能先看效果。

## 上线后怎样更新？

项目附带 `.github/workflows/deploy.yml`。按 README 在 GitHub Pages 中选择 GitHub Actions 作为来源后，推送到 `main` 会运行构建并尝试发布。

也可以直接在 GitHub 网页上创建或编辑 Markdown 文件并提交。没有在这次交付中操作你的账号；线上部署仍需你完成设置。

## 什么内容适合精选？

精选文章不一定最长，但应能说明：问题是什么，你怎样理解，做了什么验证，以及什么地方仍不确定。

每天五到十分钟留下短日志；每周挑一个真正弄懂的问题，整理成正式文章。没有新理解时，继续修订旧文章也很好，不必追求虚假的连续更新。

## 私人草稿不是“隐藏文章”

**`published: false`、不在首页展示、或者没有链接，都不是隐私保护。**

公开仓库中的 Markdown 原文与历史提交仍然可能被他人读取。敏感草稿应保存在项目外，或保存到未提交的 `private-notes/`。该文件夹默认在 `.gitignore` 中，但仍需自己检查没有误传，尤其不要通过网页上传整个私人目录。

不要发布密码、API 密钥、令牌、导师未公开结果、合作方数据或未经许可的论文全文。优先用自己的话总结，并保留原文出处。

## 求职前的检查

把“读过”“理解过”“复现过”“独立做过”区分开。只展示实际做过的工作，给代表项目补运行说明与真实结果。

把全部示例删除或隐藏，确认姓名、日期、链接和项目状态正确。主页保留两到三项最相关的作品，不要仅用文章数量来代替能力证据。

## 技术资料

- [GitHub Pages 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
- [GitHub Pages 自定义构建流程](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [在 GitHub 网页创建文件](https://docs.github.com/en/repositories/working-with-files/managing-files/creating-new-files)
- [Mistune Markdown 解析器文档](https://mistune.lepture.com/en/latest/)

公开托管服务可能有独立的访问日志与隐私规则；本模板没有添加第三方访问统计脚本。
