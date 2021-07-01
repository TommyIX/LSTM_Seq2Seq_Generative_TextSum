import requests
from bs4 import BeautifulSoup


news_url = "https://www.cbsnews.com/"   # CBS官网


# 从CBS的新闻网站中获取新闻内容
def get_news_from_CBS(news_url):
    result = []
    # 下载页面数据
    try:
        res = requests.get(news_url)   # 排除链接无法打开的情况
    except:
        return []   # 链接无法打开，则返回空列表，即没有新闻内容
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')
    # 获取新闻内容
    result.append(' '.join([p.text.strip() for p in soup.select('.content__body>p')[:]]))   # CBS新闻内容在标签p中
    return result


list_href = []   # 存放所有新闻链接


# 从CBS官网上获取所有链接
def get_links(url):
    res = requests.get(url)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')
    urls_li = soup.select("a")   # 所有超链接都在标签a中
    for url in urls_li:
        url_href = url.get('href')   # href对应的是链接本身
        list_href.append(url_href)


# 去掉那些不是新闻链接的链接
get_links(news_url)
for href in list_href:
    if 'https' not in href:   # 正确的链接前都带有http，没有http的“链接”则移出列表
        list_href.remove(href)
print(list_href)

# 从每个新闻链接中获取新闻内容存放在news_content.txt文件中
f = open('news_content.txt', 'w', encoding='utf-8')
for href in list_href:
    content = get_news_from_CBS(href)
    if content == [''] or content == []:   # 存放新闻内容的列表为空字符串或为空则不写入文件中
        pass
    else:
        f.write(str(content))
        f.write('\n\n')   # 隔一行写入一条新闻，方便后续作为模型的输入
f.close()


