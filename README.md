# 边学边记：个人学习网站起步版

这是一个适合记录论文阅读、算法理解、实验过程与短日志的个人学习网站。技术路线是 **Markdown + Python 静态生成 + GitHub Pages**。

它是一个可以持续修改的轻量起步版，不包含在线编辑后台。你平时维护 Markdown 源文件，构建脚本会把它们转换为静态网页。

## 1. 先查看网站效果

直接打开 `public/index.html` 即可查看已经生成的网站。

不要只移动 `index.html`，它依赖同目录下的 `assets/`、`notes/` 等文件。当前附带的示例文章均标记为“示例”，不代表你的真实学习经历。

也可以在项目根目录启动本地服务器：

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory public
```

然后在浏览器访问 <http://localhost:8000>。

## 2. 项目结构

- `site.yml`：网站名称、姓名、研究方向、简介和公开联系方式。
- `pages/about.md`：个人介绍。
- `pages/guide.md`：网站内显示的写作指南。
- `content/`：你的 Markdown 学习记录。
- `note-templates/`：论文、算法、项目和日志四种写作模板。
- `assets/`：样式、脚本、图片、示例代码和公开附件。
- `layouts/`：页面模板，初期通常不需要修改。
- `build.py`：静态网站生成器。
- `public/`：生成结果，下次构建时会被覆盖，不要在里面维护正文。
- `.github/workflows/deploy.yml`：推送到 `main` 后自动构建并部署。
- `tests/`：构建器的基础测试。
- `private-notes/`：可在本地自行创建，已列入 `.gitignore`，不会参与构建。

## 3. 本地生成环境

需要 Python 3.11 或更高版本。在 Windows PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe build.py
```

本项目当前已创建 `.venv` 并安装依赖，后续通常只需运行最后两条命令。

macOS 或 Linux：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python build.py
```

依赖版本固定在 `requirements.txt` 中。首次安装依赖需要连接 Python 软件源。

## 4. 修改网站信息

先编辑 `site.yml`：

```yaml
title: "边学边记"
author: "你的名字"
role: "边缘计算方向 · 研究生"
description: "记录论文阅读、算法理解与实验过程，让每一次学习都有迹可循。"
email: ""
github: ""
resume: ""
show_examples: true
```

没有准备公开的联系方式时保持空字符串。`github` 应填写完整的 HTTPS 地址；`resume` 可以填写 HTTPS 地址或 `assets/resume.pdf`。

然后修改 `pages/about.md`，用真实、准确的内容介绍自己。

## 5. 每天怎样新增记录

从 `note-templates/` 复制合适的模板到 `content/`，使用容易识别的英文文件名，例如：

```text
content/2026-09-19-my-first-note.md
```

文件开头的元数据示例：

```yaml
---
title: "今天我终于区分了计算时间与传输时间"
slug: "first-offloading-note"
kind: "journal"
date: "2026-09-19"
summary: "记下一次具体的理解变化，以及仍需验证的问题。"
tags: ["边缘计算", "任务卸载"]
featured: false
published: true
example: false
---
```

注意：

- `slug` 是网页地址中的短名，只能使用小写英文字母、数字和连字符，且不能重复。
- `kind` 只能是 `paper`、`algorithm`、`project` 或 `journal`。
- `date` 使用实际记录日期。
- `tags` 必须是 YAML 列表，并且标签中不要包含竖线字符 `|`。
- 只有 `published: true` 的文章才会生成网页。
- `featured: true` 会让文章进入首页精选区，首页最多展示三篇。
- `example: true` 表示演示内容，不计入正式记录。
- 搜索范围是标题、摘要和标签，不是全文搜索。
- 修改 Markdown 后需要重新构建；`public/` 不会自动更新。

## 6. Markdown、图片与附件

支持常见的 Markdown 标题、列表、链接、表格、代码块和脚注。HTML 标签会被转义，不会直接作为脚本执行。

当前版本没有 LaTeX 数学公式渲染、在线编辑后台、评论、全文检索或自动论文抓取功能。

图片建议放到 `assets/images/`。由于文章页面位于 `notes/` 子目录，在 Markdown 中可以这样引用：

```markdown
![说明图片内容](../assets/images/your-figure.png)
```

构建时会检查本地文件链接。优先使用自己制作或有权公开的图片和附件。

## 7. 第一次上传 GitHub

当前目录已经初始化为 Git 仓库，默认分支是 `main`。在 GitHub 创建一个空仓库后，在项目根目录运行：

```powershell
git add .
git commit -m "初始化个人学习网站"
git remote add origin 你的GitHub仓库地址
git push -u origin main
```

仓库根目录应该直接包含 `build.py`、`site.yml`、`requirements.txt`、`content/` 等内容，不要在外面额外嵌套一层目录。

`public/` 已被 `.gitignore` 排除，不需要提交；GitHub Actions 会在线重新生成它。请确认仓库中存在 `.github/workflows/deploy.yml`。

随后在 GitHub 仓库中打开 **Settings → Pages → Build and deployment → Source**，选择 **GitHub Actions**。推送到 `main` 后，工作流会执行测试、生成网页并尝试部署。

个人主页仓库通常命名为 `你的用户名.github.io`；普通项目仓库发布后通常带仓库名路径。本项目使用相对链接，两种形式均可使用。

这份项目没有替你创建 GitHub 仓库或操作账号权限。首次上线后，请在 Actions 和 Pages 页面确认部署结果。

## 8. 公开范围与隐私

**`published: false`、不在首页显示或没有网页链接，都不等于保密。**

如果使用公开 GitHub 仓库，仓库中的 Markdown 原文和历史提交仍可能被任何人看到。私人草稿应放在项目外，或保存在未提交的 `private-notes/` 中。

不要提交：

- 密码、API 密钥、访问令牌；
- 身份证号、家庭住址等个人敏感信息；
- 导师尚未公开的成果或合作方数据；
- 未经授权转载的论文全文或大段原文。

论文笔记应以自己的理解为主，并保留出处。公开前请逐项检查将要提交的文件。

## 9. 示例内容与求职真实性

项目默认包含四篇演示文章，用于展示布局。正式分享前可以删除 `content/example-*.md`，或者将 `site.yml` 中的 `show_examples` 改成 `false`。

注意：隐藏示例只会让它们不出现在生成的网站中，不会从公开仓库删除源文件。

请清楚区分“读过”“理解过”“复现过”和“独立完成过”。项目经历、技能熟练度、实验结果和日期都应按真实情况填写。

## 10. 测试与构建检查

常用命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe assets\examples\greedy.py
.\.venv\Scripts\python.exe build.py
```

生成器会检查常见元数据错误、重复 `slug` 和本地文件链接，但不会验证文章中的科学结论、外部链接是否持续有效或引用是否准确。

## 11. 建议的第一周任务

1. 修改个人介绍和网站基本信息。
2. 发布一条真实的短日志。
3. 整理一篇包含自己解释的学习笔记。
4. 删除或隐藏示例，并检查所有公开内容与链接。

先让网站服务于学习记录，再逐步调整外观和功能。

## 相关资料

- [GitHub Pages 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
- [GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [在 GitHub 网页创建文件](https://docs.github.com/en/repositories/working-with-files/managing-files/creating-new-files)
- [Mistune 文档](https://mistune.lepture.com/en/latest/)

