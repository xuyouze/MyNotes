# conda 
`进入环境` conda activate environment-name
`退出环境` conda deactivate 
`查询所有环境` conda env list
`新建环境` conda create --name your_env_name python=3.5 anaconda
`删除环境` conda remove --name your_env_name --all

# jupyter
在用之前 确保已经装了 `ipykernel`
`查询kernel` jupyter kernelspec list
`删除kernel` jupyter kernelspec remove kernelname
`增加kernel` source activate conda_env_name
python -m ipykernel install --user --name conda_env_name --display-name "display_name"

# git 
`克隆` git clone git_url
`初始化` git init
`查看状态` git status
`查看变更内容` git diff
`add 更新、新建的文件` git add .
`add 所有的文件` git add -A 
`add 更新的文件` git add -u
`add 指定文件` git add <file_name>
`移动文件` git mv <old> <new>
`删除文件` git rm <file_name>
`停止追踪，但不删除` git rm  --cached <file_name>
`提交文件` git commit -m 'commit message'
`撤销已经add的文件` git head reset <file_name>
`撤销未提交的文件修改内容` git check <file_name>
`查看提交历史` git log <file_name>
`上传代码` git push <remote> <branch>
`下载代码` git pull <remote> <branch>
`合并冲突` git merge
`图形化冲突解决`git mergetool 


# markdown
## head

## 引用文字
In the words of Abraham Lincoln:
> Pardon my French

## 文字效果
`加粗` **hello** 或者 __hello__
`倾斜` *hello* 或者 _hello_
`删除` ~~This was mistaken text~~
`加粗倾斜`	**_hello_**
 `<hello>`

## 引用代码
```javascript
function fancyAlert(arg) {
  if(arg) {
    $.facebox({div:'#foo'})
  }
}
```
##链接 
`链接`[GitHub Pages](https://pages.github.com/).
`图片`![GitHub Logo](../images/logo.ipg)


## 列表
- George Washington
- John Adams
- Thomas Jefferson

## 任务列表
- [x] Finish my changes
- [ ] Push my commits to GitHub
- [ ] Open a pull request

## 表格
First Header | Second Header
------------ | -------------
Content from cell 1 | Content from cell 2
Content in the first column | Content in the second column


# 服务器上相关操作

`查询cuda` nvcc  --version
`查询GPU` nvidia-smi
`每十秒输出` watch -n 10 nvidia-smi

`ssh登陆` ssh username@host_ip 
`scp传输` scp /path/local_filename username@servername:/path  
`查询文件大小` du -sh filename
`传输` scp -r /Users/if-pc/Documents/paper/facial\ attribute/ xuyouze@192.168.1.4:/home/xuyouze/
`复制` cp -ifr fromdir todir
