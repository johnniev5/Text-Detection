### 项目简介
```
data目录存放训练用的数据和向量化过程中用到的index对应关系的转换
models目录主要存放的是模型文件
config目录存放的是模型和监控配置文件等
output目录作为文件输入的结果输出
train.py为训练脚本
```

### 执行环境

#### python包
运行环境所依赖的python包都在requrements.txt中可找到，可直接按此文件进行安装即可。

#### tensorflow serving
这里默认使用的是tensorflow提供的restful api，且以容器化运行较为便捷

```
# 项目所在目录，包括模型和配置目录等信息
export URL_HOME=/home/johnnie/桌面/url/url/

# 拉取docker镜像
sudo docker pull tensorflow/serving
sudo docker run -d -p 8501:8501 -v "${URL_HOME}/models:/models" -v "${URL_HOME}/config:/config" tensorflow/serving --model_config_file=/config/models.config --monitoring_config_file=/config/monitor.config --enable_batching=true --batching_parameters_file=/config/batching.config --model_config_file_poll_wait_seconds=10 --file_system_poll_wait_seconds=10 --allow_version_labels_for_unavailable_models=true
```

### 测试方法

```
# 直接基于python环境测试
python main.py --input_text="#plunaiscool #transrights #transrightsarehumanrights you should follow this really cool person on twittr (not gettr since aids site) https://americanmilitarynews.com/2021/07/trump-slams-ridiculous-milley-coup-allegations-says-general-trying-to-curry-favor-with-radical-left-and-more/?utm_campaign=DailyEmails&utm_source=AM_Email&utm_medium=email&utm_source=Master_List&utm_campaign=3238847e44-EMAIL_CAMPAIGN_2021_07_15_10_29&utm_medi http://turnmeon.club/ https://hotmen-club.com/?u=y0fw2k7&o=2nfpgbn http://gettr.com@fwme.eu/19wxq https://twitter.com/harrylitman/status/1412802184027074560?s=21@trumpteam https://www.biblegateway.com/passage/?search=Psalm%20119:30&version=nkjv&cm_mmc=ExactTarget-_-VOTDBGNLT-_-ponchr@sbcglobal.net-_-reference-verse https://bit.ly/3yc084n https://ix.sk/AVj1l"

--input_text为输入的文本内容，即为用户的posts和comments，文本里可能会包含多个url链接，链接之间默认以空格分隔，即可进行文本内多个url的判断，这也是符合实际需求的。

输出结果如下：

{'result': [{'orginal_url': 'https://americanmilitarynews.com/2021/07/trump-slams-ridiculous-milley-coup-allegations-says-general-trying-to-curry-favor-with-radical-left-and-more/?utm_campaign=DailyEmails&utm_source=AM_Email&utm_medium=email&utm_source=Master_List&utm_campaign=3238847e44-EMAIL_CAMPAIGN_2021_07_15_10_29&utm_medi', 'long_url': 'https://americanmilitarynews.com/2021/07/trump-slams-ridiculous-milley-coup-allegations-says-general-trying-to-curry-favor-with-radical-left-and-more/?utm_campaign=DailyEmails&utm_source=AM_Email&utm_medium=email&utm_source=Master_List&utm_campaign=3238847e44-EMAIL_CAMPAIGN_2021_07_15_10_29&utm_medi', 'pred': '白', 'score': 1.0}, {'orginal_url': 'http://turnmeon.club/', 'long_url': 'http://turnmeon.club/', 'pred': '黑', 'score': 1.0}, {'orginal_url': 'https://hotmen-club.com/?u=y0fw2k7&o=2nfpgbn', 'long_url': 'https://hotmen-club.com/?u=y0fw2k7&o=2nfpgbn', 'pred': '黑', 'score': 1.0}, {'orginal_url': 'http://gettr.com@fwme.eu/19wxq', 'long_url': 'http://fwme.eu', 'pred': '白', 'score': 1.0}, {'orginal_url': 'https://twitter.com/harrylitman/status/1412802184027074560?s=21@trumpteam', 'long_url': 'https://twitter.com/harrylitman/status/1412802184027074560?s=21@trumpteam', 'pred': '白', 'score': 1.0}, {'orginal_url': 'https://www.biblegateway.com/passage/?search=Psalm%20119:30&version=nkjv&cm_mmc=ExactTarget-_-VOTDBGNLT-_-ponchr@sbcglobal.net-_-reference-verse', 'long_url': 'https://www.biblegateway.com/passage/?search=Psalm%20119:30&version=nkjv&cm_mmc=ExactTarget-_-VOTDBGNLT-_-ponchr@sbcglobal.net-_-reference-verse', 'pred': '白', 'score': 1.0}, {'orginal_url': 'https://bit.ly/3yc084n', 'long_url': 'https://patriotpoweredspecials.com/order-376942091614720517756?hop=usamagazin', 'pred': '白', 'score': 1.0}, {'orginal_url': 'https://ix.sk/AVj1l', 'long_url': 'http://ttvu.2track.info', 'pred': '白', 'score': 1.0}]}

# 基于curl请求测试
instances=$(python data2vec.py "#plunaiscool #transrights #transrightsarehumanrights you should follow this really cool person on twittr (not gettr since aids site) https://americanmilitarynews.com/2021/07/trump-slams-ridiculous-milley-coup-allegations-says-general-trying-to-curry-favor-with-radical-left-and-more/?utm_campaign=DailyEmails&utm_source=AM_Email&utm_medium=email&utm_source=Master_List&utm_campaign=3238847e44-EMAIL_CAMPAIGN_2021_07_15_10_29&utm_medi http://turnmeon.club/ https://hotmen-club.com/?u=y0fw2k7&o=2nfpgbn http://gettr.com@fwme.eu/19wxq https://twitter.com/harrylitman/status/1412802184027074560?s=21@trumpteam https://www.biblegateway.com/passage/?search=Psalm%20119:30&version=nkjv&cm_mmc=ExactTarget-_-VOTDBGNLT-_-ponchr@sbcglobal.net-_-reference-verse https://bit.ly/3yc084n https://ix.sk/AVj1l")

curl -v -d "{\"instances\": $instances}" -X POST http://localhost:8501/v1/models/url:predict

输出结果如下：
{
	"predictions": [[0.0], [1.0], [1.0], [1.40770277e-11], [4.91908026e-28], [0.0], [1.12264252e-21], [2.03563972e-14]
}
此结果需要进行一些处理，具体处理格式可参考main.py中的main函数。    


# 进行ab压力测试
ab -n 10000 -c 100 -p 'post.txt' -T 'application/json' 'http://localhost:8501/v1/models/url:predict'

This is ApacheBench, Version 2.3 <$Revision: 1879490 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking localhost (be patient)
Completed 1000 requests
Completed 2000 requests
Completed 3000 requests
Completed 4000 requests
Completed 5000 requests
Completed 6000 requests
Completed 7000 requests
Completed 8000 requests
Completed 9000 requests
Completed 10000 requests
Finished 10000 requests


Server Software:        
Server Hostname:        localhost
Server Port:            8501

Document Path:          /v1/models/url:predict
Document Length:        83 bytes

Concurrency Level:      100
Time taken for tests:   0.728 seconds
Complete requests:      10000
Failed requests:        0
Non-2xx responses:      10000
Total transferred:      1430000 bytes
Total body sent:        160800000
HTML transferred:       830000 bytes
Requests per second:    13729.72 [#/sec] (mean)
Time per request:       7.283 [ms] (mean)
Time per request:       0.073 [ms] (mean, across all concurrent requests)
Transfer rate:          1917.33 [Kbytes/sec] received
                        215599.50 kb/s sent
                        217516.84 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    1   0.5      1       4
Processing:     1    7   7.4      5     100
Waiting:        1    6   7.4      5     100
Total:          2    7   7.4      6     101

Percentage of the requests served within a certain time (ms)
  50%      6
  66%      7
  75%      7
  80%      8
  90%      9
  95%     11
  98%     29
  99%     49
 100%    101 (longest request)
```
