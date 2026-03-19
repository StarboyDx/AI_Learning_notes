import requests

url='http://localhost:5000/sentiment'
sentence='感觉还不错，液晶电视，配有电脑。位置很好,'
res=requests.get(url,params={'sentence':sentence})#发送请求
print(res.json())